#!/usr/bin/env python3

import asyncio
import json
import time
import logging
import re
from typing import Dict, List, Optional

import numpy as np
from pybit.unified_trading import HTTP
import google.generativeai as genai
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate, QFT
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_aer import Aer

# --- Configuration ---
BYBIT_API_KEY = "efPuWmUveGLBhsmKB3"  # Replace with new key
BYBIT_SECRET_KEY = "yarPSLw5ZozT2ErRzKMoF7R3LQix7TaQcb9c"  # Replace with secret
IBM_QUANTUM_API_TOKEN = "334d70c5d25a2b538142a228c1c9b30d02496429469596a62c3e158198caf82536d63b8ce7722ee7befe404626dac86c7c7bea212ee4c535dbfc516bf1c3963f"
GEMINI_API_KEY = "AIzaSyB6EvC7ynKiDuJKDZIOZxWj8VK8L5lFVrQ"
QUANTUM_SHOTS = 4096
PREDICTION_HORIZON_MIN = 30
PREDICTION_HORIZON_MAX = 60
RISK_THRESHOLD = 0.002
MAX_LEVERAGE = 20
POLL_INTERVAL = 2
CONSISTENCY_CHECKS = 7

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)

# --- Initialize Services ---
service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_QUANTUM_API_TOKEN)
logger.info("Qiskit Runtime initialized")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
logger.info("Gemini API initialized")
bybit = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_SECRET_KEY, testnet=False)

# --- Bybit API Functions ---
async def get_account_balances() -> tuple[float, Dict[str, float]]:
    account_types = ["SPOT", "UNIFIED", "FUNDING"]
    total_usdt = 0.0
    coin_balances = {}
    for acc_type in account_types:
        for _ in range(3):  # Retry logic
            try:
                wallet = bybit.get_wallet_balance(accountType=acc_type)
                logger.info(f"Raw wallet response ({acc_type}): {wallet}")
                if wallet["retCode"] == 0:
                    total_usdt += float(wallet["result"]["list"][0]["totalEquity"])
                coins = bybit.get_coin_balance(accountType=acc_type)
                logger.info(f"Raw coin response ({acc_type}): {coins}")
                if coins["retCode"] == 0:
                    for coin in coins["result"]["balance"]:
                        coin_balances[coin["coin"]] = coin_balances.get(coin["coin"], 0) + float(coin["walletBalance"])
                break
            except Exception as e:
                logger.error(f"Balance fetch failed for {acc_type}: {e}")
                await asyncio.sleep(1)
    logger.info(f"Total balance: {total_usdt} USDT, Coins: {coin_balances}")
    return total_usdt, coin_balances

async def convert_to_usdt(coin: str, qty: float, crypto_data: Dict) -> float:
    symbol = f"{coin}USDT"
    try:
        order = bybit.place_order(category="spot", symbol=symbol, side="Sell", orderType="Market", qty=str(qty))
        if order["retCode"] == 0:
            profit = qty * crypto_data.get(symbol, {}).get("lastPrice", 0)
            logger.info(f"Converted {qty} {coin} to {profit} USDT")
            return profit
        logger.error(f"Conversion failed: {order['retMsg']}")
        return 0.0
    except Exception as e:
        logger.error(f"Conversion error for {coin}: {e}")
        return 0.0

async def place_order(symbol: str, side: str, qty: float, price: float, stop_loss: float) -> bool:
    try:
        order = bybit.place_order(
            category="spot", symbol=symbol, side=side.capitalize(),
            orderType="Limit", qty=str(qty), price=str(price), timeInForce="GTC"
        )
        if order["retCode"] == 0:
            logger.info(f"Order placed: {side} {qty} {symbol} at {price}")
            bybit.set_trading_stop(category="spot", symbol=symbol, stopLoss=str(stop_loss))
            logger.info(f"Stop-loss set at {stop_loss} for {symbol}")
            return True
        logger.error(f"Order failed: {order['retMsg']}")
        return False
    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        return False

async def get_live_crypto_data(symbols: List[str]) -> Dict:
    try:
        data = {}
        for symbol in symbols:
            kline = bybit.get_kline(category="spot", symbol=symbol, interval="1", limit=60)
            orderbook = bybit.get_orderbook(category="spot", symbol=symbol)
            if kline["retCode"] == 0 and orderbook["retCode"] == 0:
                last_kline = kline["result"]["list"][0]
                data[symbol] = {
                    "lastPrice": float(last_kline[4]),
                    "volume24h": float(last_kline[5]),
                    "highPrice24h": float(last_kline[2]),
                    "lowPrice24h": float(last_kline[3]),
                    "bid": float(orderbook["result"]["b"][0][0]),
                    "ask": float(orderbook["result"]["a"][0][0])
                }
        logger.info(f"Live Crypto Data: {data}")
        return data
    except Exception as e:
        logger.error(f"Live data failed: {e}")
        return {}

# --- Quantum Addons ---
def quantum_momentum_circuit(crypto_data: Dict, trade_symbols: List[str]) -> QuantumCircuit:
    num_qubits = min(len(trade_symbols), 6)
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(qr, cr)
    for i, symbol in enumerate(trade_symbols[:num_qubits]):
        momentum = (crypto_data[symbol]["lastPrice"] - crypto_data[symbol]["lowPrice24h"]) / crypto_data[symbol]["lastPrice"]
        qc.h(qr[i])
        qc.rz(momentum * np.pi, qr[i])
    qc.append(QFT(num_qubits), qr)
    qc.measure(qr, cr)
    return qc

async def quantum_momentum(crypto_data: Dict, trade_symbols: List[str]) -> Dict:
    logger.info("Quantum momentum analysis")
    qc = quantum_momentum_circuit(crypto_data, trade_symbols)
    try:
        sampler = Sampler(service.backend("ibm_brisbane"))
        job = sampler.run(qc, shots=QUANTUM_SHOTS)
        result = job.result()
        counts = result.quasi_dists[0].binary_probabilities()
    except Exception as e:
        logger.warning(f"IBM Quantum failed: {e}, falling back to Aer")
        backend = Aer.get_backend("qasm_simulator")
        job = execute(qc, backend, shots=QUANTUM_SHOTS)
        counts = job.result().get_counts()

    momentum = {}
    total_shots = sum(counts.values())
    for state, count in counts.items():
        prob = count / total_shots
        for i, bit in enumerate(reversed(state)):
            symbol = trade_symbols[i] if i < len(trade_symbols) else None
            if symbol:
                momentum[symbol] = momentum.get(symbol, 0) + (int(bit) * prob)
    logger.info(f"Quantum Momentum: {momentum}")
    return momentum

def quantum_risk_shield(crypto_data: Dict, params: Dict, trade_symbols: List[str]) -> QuantumCircuit:
    num_qubits = min(len(trade_symbols), 6)
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(qr, cr)
    param_values = list(params.values())  # Convert dict to list
    coupling_map = CouplingMap.from_line(num_qubits)
    hamiltonian = generate_xyz_hamiltonian(coupling_map, coupling_constants=(0.1, 0.2, 0.3))
    qc.append(PauliEvolutionGate(hamiltonian, time=1.0), qr)
    for i, param in enumerate(param_values[:num_qubits]):
        volatility = (crypto_data.get(trade_symbols[i], {}).get("highPrice24h", 1.0) - 
                      crypto_data.get(trade_symbols[i], {}).get("lowPrice24h", 1.0)) / 100
        qc.rx(param * volatility, qr[i])
        if i > 0:
            qc.cx(qr[i-1], qr[i])
    qc.measure(qr, cr)
    return qc

async def quantum_risk_assessment(crypto_data: Dict, params: Dict, trade_symbols: List[str]) -> Dict:
    logger.info("Quantum risk assessment")
    qc = quantum_risk_shield(crypto_data, params, trade_symbols)
    try:
        sampler = Sampler(service.backend("ibm_brisbane"))
        job = sampler.run(qc, shots=QUANTUM_SHOTS)
        result = job.result()
        counts = result.quasi_dists[0].binary_probabilities()
    except Exception as e:
        logger.warning(f"IBM Quantum failed: {e}, falling back to Aer")
        backend = Aer.get_backend("qasm_simulator")
        job = execute(qc, backend, shots=QUANTUM_SHOTS)
        counts = job.result().get_counts()

    risks = {}
    total_shots = sum(counts.values())
    for state, count in counts.items():
        prob = count / total_shots
        for i, bit in enumerate(reversed(state)):
            symbol = trade_symbols[i] if i < len(trade_symbols) else None
            if symbol:
                risks[symbol] = risks.get(symbol, 0) + (int(bit) * prob)
    logger.info(f"Quantum Risk Scores: {risks}")
    return risks

def quantum_profit_amplifier(crypto_data: Dict, params: Dict, trade_symbols: List[str]) -> QuantumCircuit:
    num_qubits = min(len(trade_symbols), 6)
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(qr, cr)
    param_values = list(params.values())
    for i, symbol in enumerate(trade_symbols[:num_qubits]):
        qc.h(qr[i])
        qc.rz(param_values[i] * crypto_data[symbol]["lastPrice"] / 100, qr[i])
    qc.append(PauliEvolutionGate(SparsePauliOp(["Z" * num_qubits]), time=1.0), qr)
    qc.measure(qr, cr)
    return qc

async def quantum_prediction(crypto_data: Dict, params: Dict, trade_symbols: List[str]) -> Dict:
    logger.info("Quantum prediction")
    qc = quantum_profit_amplifier(crypto_data, params, trade_symbols)
    try:
        sampler = Sampler(service.backend("ibm_brisbane"))
        job = sampler.run(qc, shots=QUANTUM_SHOTS)
        result = job.result()
        counts = result.quasi_dists[0].binary_probabilities()
    except Exception as e:
        logger.warning(f"IBM Quantum failed: {e}, falling back to Aer")
        backend = Aer.get_backend("qasm_simulator")
        job = execute(qc, backend, shots=QUANTUM_SHOTS)
        counts = job.result().get_counts()

    predictions = {symbol: [] for symbol in trade_symbols[:6]}
    for _ in range(CONSISTENCY_CHECKS):
        total_shots = sum(counts.values())
        temp_preds = {}
        for state, count in counts.items():
            prob = count / total_shots
            for i, bit in enumerate(reversed(state)):
                symbol = trade_symbols[i] if i < len(trade_symbols) else None
                if symbol:
                    temp_preds[symbol] = temp_preds.get(symbol, 0) + (int(bit) * prob)
        for symbol in temp_preds:
            predictions[symbol].append(temp_preds[symbol])

    final_preds = {}
    for symbol, scores in predictions.items():
        avg_score = np.mean(scores)
        std_dev = np.std(scores)
        current_price = crypto_data[symbol]["lastPrice"]
        high_pred = current_price * (1 + avg_score * 0.1)
        low_pred = current_price * (1 - avg_score * 0.1)
        final_preds[symbol] = {"high": high_pred, "low": low_pred, "confidence": 1 - std_dev}
    logger.info(f"Quantum Predictions: {final_preds}")
    return final_preds

# --- Trading Logic ---
async def quantum_convert_to_usdt(crypto_data: Dict, coin_balances: Dict[str, float]) -> float:
    total_usdt = coin_balances.get("USDT", 0.0)
    symbols = [f"{coin}USDT" for coin in coin_balances if coin != "USDT" and coin_balances[coin] > 0]
    if not symbols:
        return total_usdt

    quantum_params = {"p" + str(i): v for i, v in enumerate(np.random.uniform(0, 2 * np.pi, min(len(symbols), 6)))}
    predictions = await quantum_prediction(crypto_data, quantum_params, symbols)
    
    for coin, qty in coin_balances.items():
        if coin == "USDT" or qty <= 0:
            continue
        symbol = f"{coin}USDT"
        if symbol in predictions and predictions[symbol]["confidence"] > 0.9:
            price = predictions[symbol]["high"]
            if await place_order(symbol, "Sell", qty, price, price * 0.99):
                profit = qty * price
                total_usdt += profit
                logger.info(f"Quantum-converted {qty} {coin} to {profit} USDT at {price}")
        else:
            total_usdt += await convert_to_usdt(coin, qty, crypto_data)
    logger.info(f"Total USDT after conversions: {total_usdt}")
    return total_usdt

async def quantum_spot_grid(symbol: str, funds: float, predictions: Dict) -> List[Dict]:
    pred = predictions[symbol]
    step = (pred["high"] - pred["low"]) / 4
    qty_per_grid = funds / 5 / pred["low"]
    stop_loss = pred["low"] * 0.99

    trades = []
    for i in range(5):
        buy_price = pred["low"] + (i * step)
        sell_price = buy_price * 1.1  # 10% profit target
        trades.append({"symbol": symbol, "side": "Buy", "qty": qty_per_grid, "price": buy_price, "stop_loss": stop_loss})
        trades.append({"symbol": symbol, "side": "Sell", "qty": qty_per_grid, "price": sell_price})
    logger.info(f"Quantum spot grid for {symbol}: {trades}")
    return trades

async def execute_quantum_trades(crypto_data: Dict, predictions: Dict, risks: Dict, momentum: Dict, trades: List[Dict], balance: float) -> float:
    logger.info("Executing quantum trades")
    leverage = min(MAX_LEVERAGE, 1 + (sum(momentum.values()) / len(momentum)) * 10)
    available_funds = balance * leverage
    profit = 0.0
    for trade in trades:
        symbol = trade["symbol"]
        side = trade["side"]
        qty = float(trade["qty"])
        price = float(trade["price"])
        stop_loss = trade["stop_loss"]
        cost = qty * price
        if cost > available_funds:
            qty = available_funds / price
            logger.info(f"Adjusted qty for {symbol} to {qty}")

        pred = predictions[symbol]
        risk_score = risks[symbol]
        if risk_score > RISK_THRESHOLD:
            logger.warning(f"Skipping {symbol}: Risk {risk_score} > {RISK_THRESHOLD}")
            continue
        if pred["confidence"] < 0.9:
            logger.warning(f"Skipping {symbol}: Confidence {pred['confidence']} < 0.9")
            continue

        if await place_order(symbol, side, qty, price, stop_loss):
            current_price = crypto_data[symbol]["lastPrice"]
            profit += (price - current_price) * qty if side.lower() == "sell" else (current_price - price) * qty
            available_funds -= cost
            logger.info(f"Profit from {symbol}: {profit} USDT")
    return profit

# --- Gemini Integration ---
async def get_gemini_insights(crypto_data: Dict, total_usdt: float) -> Dict:
    logger.info("Fetching Gemini insights")
    data_summary = "\n".join(f"{s}: Last={d['lastPrice']}, Vol={d['volume24h']}" for s, d in crypto_data.items())
    quantum_params = {"parameter" + str(i): float(v) for i, v in enumerate(np.random.uniform(0, 2 * np.pi, 6), 1)}
    prompt = (
        f"Quantum trader: Turn {total_usdt} USDT into 50000 USDT fast. Max leverage: {MAX_LEVERAGE}x. "
        f"Live data:\n{data_summary}\n"
        f"Predict top tokens 30-60s ahead for high-profit trades with stop-losses. "
        f"Return JSON: 5+ trades (buy/sell, symbol, qty, price, stop_loss) and 6 quantum params."
    )
    try:
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        json_str = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL).group(1)
        result = json.loads(json_str)
        logger.info(f"Gemini insights: {result}")
        return result
    except Exception as e:
        logger.error(f"Gemini failed: {e}")
        return {"trades": [], "quantum_params": quantum_params}

# --- Main Execution ---
async def trading_workflow():
    logger.info("Starting trading workflow")
    total_usdt, coin_balances = await get_account_balances()
    if total_usdt < 5:
        logger.error("Insufficient funds; need at least 5 USDT")
        return {"status": "No funds"}

    trade_symbols = ["PARTIUSDT", "MAVIAUSDT", "FIREUSDT", "MOVEUSDT", "XARUSDT"]
    crypto_data = await get_live_crypto_data(trade_symbols)
    if not crypto_data:
        return {"status": "No data"}

    total_usdt = await quantum_convert_to_usdt(crypto_data, coin_balances)
    gemini_insights = await get_gemini_insights(crypto_data, total_usdt)
    trade_symbols = list(set(trade_symbols + [t["symbol"] for t in gemini_insights["trades"]]))

    risks = await quantum_risk_assessment(crypto_data, gemini_insights["quantum_params"], trade_symbols)
    momentum = await quantum_momentum(crypto_data, trade_symbols)
    predictions = await quantum_prediction(crypto_data, gemini_insights["quantum_params"], trade_symbols)
    if not all([risks, momentum, predictions]):
        return {"status": "Quantum failure"}

    quantum_profit = await execute_quantum_trades(crypto_data, predictions, risks, momentum, gemini_insights["trades"], total_usdt)
    grid_symbol = max(momentum, key=momentum.get)
    grid_trades = await quantum_spot_grid(grid_symbol, total_usdt / 2, predictions)
    grid_profit = await execute_quantum_trades(crypto_data, predictions, risks, momentum, grid_trades, total_usdt / 2)

    total_profit = quantum_profit + grid_profit
    result = {"status": "Cycle completed", "balance": total_usdt + total_profit, "profit": total_profit}
    logger.info(f"Workflow completed: {result}")
    return result

async def main():
    logger.info("Starting main loop")
    while True:
        result = await trading_workflow()
        with open("trading_results.json", "w") as f:
            json.dump(result, f)
        if result.get("balance", 0) >= 50000:
            logger.info("Target of 50,000 USDT reached!")
            break
        await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
