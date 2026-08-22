"""
A股数据获取 + 本地缓存

功能:
  1. 从 akshare 获取A股分钟级K线数据
  2. 本地缓存为 CSV，避免重复请求
  3. 增量更新: 只拉新增数据
  4. 统一加载接口: 从本地缓存读取

目录结构:
  data/cache/
    000001.csv    ← 平安银行 1min K线
    000002.csv    ← 万科A
    ...
    _meta.json    ← 缓存元信息 (最后更新时间等)
"""
import os
import sys
import io
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

# Cache directory
CACHE_DIR = Path(__file__).parent / "cache"
META_FILE = CACHE_DIR / "_meta.json"


# ============================================================
# 股票池
# ============================================================
# 默认关注的A股 (流动性好、代表性强)
DEFAULT_STOCKS = [
    "000001",  # 平安银行
    "000002",  # 万科A
    "000063",  # 中兴通讯
    "000333",  # 美的集团
    "000651",  # 格力电器
    "000858",  # 五粮液
    "002594",  # 比亚迪
    "600000",  # 浦发银行
    "600009",  # 上海机场
    "600036",  # 招商银行
    "600276",  # 恒瑞医药
    "600519",  # 贵州茅台
    "600887",  # 伊利股份
    "601318",  # 中国平安
    "601888",  # 中国中免
]


# ============================================================
# 数据获取
# ============================================================
def fetch_stock_minute(code: str, period: str = "1",
                       start_date: str = None,
                       end_date: str = None) -> Optional[pd.DataFrame]:
    """
    获取单只股票的分钟级K线
    
    Args:
      code: 股票代码 (如 "000001")
      period: "1"=1分钟, "5"=5分钟, "15"=15分钟, "30"=30分钟, "60"=60分钟
      start_date: 开始日期 "YYYY-MM-DD"
      end_date: 结束日期 "YYYY-MM-DD"
    
    Returns:
      DataFrame with columns: datetime, open, high, low, close, volume, amount
    """
    import akshare as ak
    
    try:
        # akshare 分钟级K线接口
        df = ak.stock_zh_a_hist_min_em(
            symbol=code,
            period=period,
            start_date=start_date or "2024-01-01 09:30:00",
            end_date=end_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            adjust="qfq",  # 前复权
        )
        
        if df is None or df.empty:
            print(f"  [WARN] {code}: 无数据")
            return None
        
        # 统一列名
        col_map = {
            '时间': 'datetime',
            '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close',
            '成交量': 'volume', '成交额': 'amount',
        }
        df = df.rename(columns=col_map)
        
        # 保留需要的列
        keep_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']
        df = df[[c for c in keep_cols if c in df.columns]]
        
        # 转换类型
        df['datetime'] = pd.to_datetime(df['datetime'])
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        
        # 去重
        df = df.drop_duplicates(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"  [ERROR] {code}: {e}")
        return None


def fetch_stock_daily_batch(codes: List[str],
                            start_date: str = "2023-01-01",
                            end_date: str = None,
                            frequency: str = "5") -> dict:
    """
    批量获取K线数据 (单次登录, baostock)
    
    Args:
      frequency: "d"=日线, "5"=5分钟, "15"=15分钟, "30"=30分钟, "60"=60分钟
    
    Returns:
      dict: {code: DataFrame}
    """
    import baostock as bs
    
    lg = bs.login()
    if lg.error_code != '0':
        print(f"  baostock login failed: {lg.error_msg}")
        return {}
    
    results = {}
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    
    for code in codes:
        if '.' not in code:
            prefix = 'sh' if code.startswith('6') else 'sz'
            bs_code = f"{prefix}.{code}"
        else:
            bs_code = code
        
        try:
            if frequency == "d":
                fields = "date,open,high,low,close,volume,amount"
            else:
                fields = "date,time,open,high,low,close,volume,amount"
            
            rs = bs.query_history_k_data_plus(
                bs_code, fields,
                start_date=start_date, end_date=end_date,
                frequency=frequency, adjustflag="2",
            )
            
            data_list = []
            while (rs.error_code == '0') and rs.next():
                data_list.append(rs.get_row_data())
            
            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                
                if frequency == "d":
                    df = df.rename(columns={'date': 'datetime'})
                    df['datetime'] = pd.to_datetime(df['datetime'])
                else:
                    # 分钟级: 合并 date + time
                    df['datetime'] = pd.to_datetime(
                        df['date'] + ' ' + df['time'].str[:2] + ':' + df['time'].str[2:4],
                        format='mixed'
                    )
                    df = df.drop(columns=['time'], errors='ignore')
                
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna(subset=['close']).reset_index(drop=True)
                results[code] = df
        except Exception as e:
            print(f"  [WARN] {code}: {e}")
    
    bs.logout()
    return results


# ============================================================
# 本地缓存管理
# ============================================================
def _load_meta() -> dict:
    """加载缓存元信息"""
    if META_FILE.exists():
        with open(META_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_meta(meta: dict):
    """保存缓存元信息"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)


def _is_fresh(code: str, max_age_hours: int = 4) -> bool:
    """检查缓存是否新鲜"""
    meta = _load_meta()
    if code not in meta:
        return False
    last_update = datetime.fromisoformat(meta[code].get('updated_at', '2000-01-01'))
    return (datetime.now() - last_update).total_seconds() < max_age_hours * 3600


def update_cache(stocks: List[str] = None, period: str = "5",
                 force: bool = False, max_age_hours: int = 4,
                 start_date: str = "2023-01-01"):
    """
    更新本地缓存 (baostock, 支持分钟级)
    
    Args:
      period: "d"=日线, "5"=5分钟, "15"=15分钟, "30"=30分钟, "60"=60分钟
    """
    if stocks is None:
        stocks = DEFAULT_STOCKS
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = _load_meta()
    
    freq_name = {"d": "日线", "5": "5分钟", "15": "15分钟", "30": "30分钟", "60": "60分钟"}.get(period, period)
    print(f"更新缓存 ({len(stocks)}只股票, {freq_name})...")
    
    data = fetch_stock_daily_batch(stocks, start_date=start_date, frequency=period)
    
    success = 0
    skipped = 0
    failed = 0
    
    for code in stocks:
        csv_path = CACHE_DIR / f"{code}_{period}min.csv" if period != "d" else CACHE_DIR / f"{code}.csv"
        
        if not force and csv_path.exists() and _is_fresh(code, max_age_hours):
            skipped += 1
            continue
        
        if code in data and not data[code].empty:
            df = data[code]
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            meta[code] = {
                'updated_at': datetime.now().isoformat(),
                'rows': len(df),
                'start': str(df['datetime'].min()),
                'end': str(df['datetime'].max()),
                'period': freq_name,
                'csv_path': str(csv_path),
            }
            success += 1
            print(f"  {code} ok {len(df)}条 ({df['datetime'].min().date()} ~ {df['datetime'].max().date()})")
        else:
            failed += 1
            print(f"  {code} FAIL")
    
    _save_meta(meta)
    print(f"\n完成: 成功={success} 跳过={skipped} 失败={failed}")
    return meta


# ============================================================
# 从本地缓存加载
# ============================================================
def load_stock(code: str) -> Optional[pd.DataFrame]:
    """从本地缓存加载单只股票"""
    csv_path = CACHE_DIR / f"{code}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    return df


def load_all(stocks: List[str] = None) -> dict:
    """加载所有缓存的股票数据"""
    if stocks is None:
        stocks = DEFAULT_STOCKS
    
    data = {}
    for code in stocks:
        df = load_stock(code)
        if df is not None and not df.empty:
            data[code] = df
    
    return data


def get_cache_info() -> pd.DataFrame:
    """查看缓存状态"""
    meta = _load_meta()
    if not meta:
        print("缓存为空")
        return pd.DataFrame()
    
    rows = []
    for code, info in meta.items():
        if code.startswith('_'):
            continue
        rows.append({
            'code': code,
            'rows': info.get('rows', 0),
            'period': info.get('period', '?'),
            'start': info.get('start', '?'),
            'end': info.get('end', '?'),
            'updated': info.get('updated_at', '?')[:19],
        })
    
    df = pd.DataFrame(rows)
    print(f"\n缓存状态 ({len(df)}只股票):")
    print(df.to_string(index=False))
    return df


# ============================================================
# 统一特征提取接口
# ============================================================
def prepare_features(code: str, history_len: int = 60) -> Optional[np.ndarray]:
    """
    从本地缓存加载并提取特征
    
    Returns:
      features: [T, n_features] numpy array
    """
    df = load_stock(code)
    if df is None or len(df) < history_len + 30:
        return None
    
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    
    T = len(close)
    features = []
    
    # 1. 收益率
    ret = np.zeros(T)
    ret[1:] = (close[1:] - close[:-1]) / (np.abs(close[:-1]) + 1e-8)
    features.append(ret)
    
    # 2. 波动率 (5期/20期)
    for w in [5, 20]:
        vol = np.zeros(T)
        for t in range(w, T):
            vol[t] = np.std(ret[t-w+1:t+1])
        features.append(vol)
    
    # 3. 均线偏离 (5/10/20/60期)
    for p in [5, 10, 20, 60]:
        ma = np.zeros(T)
        for t in range(p-1, T):
            ma[t] = np.mean(close[t-p+1:t+1])
        features.append((close - ma) / (np.abs(ma) + 1e-8))
    
    # 4. RSI (14)
    rsi = np.full(T, 0.5)
    for t in range(1, T):
        delta = close[t] - close[t-1]
        gain = max(delta, 0)
        loss = max(-delta, 0)
        if t >= 14:
            avg_g = np.mean([max(close[k]-close[k-1], 0) for k in range(t-13, t+1)])
            avg_l = np.mean([max(close[k-1]-close[k], 0) for k in range(t-13, t+1)]) + 1e-8
            rsi[t] = avg_g / (avg_g + avg_l)
    features.append(rsi)
    
    # 5. 布林带位置
    bb = np.zeros(T)
    for t in range(19, T):
        ma = np.mean(close[t-19:t+1])
        std = np.std(close[t-19:t+1]) + 1e-8
        bb[t] = (close[t] - ma) / (2 * std)
    features.append(bb)
    
    # 6. 成交量比率 (当前/20期均量)
    vol_avg = np.zeros(T)
    for t in range(19, T):
        vol_avg[t] = np.mean(volume[t-19:t+1])
    features.append(volume / (vol_avg + 1e-8))
    
    # 7. 振幅
    features.append((high - low) / (np.abs(close) + 1e-8))
    
    # 8. 收盘位置
    features.append((close - low) / (high - low + 1e-8))
    
    result = np.stack(features, axis=-1).astype(np.float32)
    
    # 填充 NaN
    result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    result = np.clip(result, -10, 10)
    
    return result


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    import argparse
    parser = argparse.ArgumentParser(description="A股数据缓存管理")
    parser.add_argument("action", choices=["update", "info", "test"],
                        help="update=更新缓存, info=查看状态, test=测试加载")
    parser.add_argument("--stocks", nargs="*", help="股票代码列表")
    parser.add_argument("--force", action="store_true", help="强制刷新")
    parser.add_argument("--period", default="1", help="K线周期 (1/5/15/30/60)")
    args = parser.parse_args()
    
    if args.action == "update":
        update_cache(stocks=args.stocks, period=args.period, force=args.force)
    elif args.action == "info":
        get_cache_info()
    elif args.action == "test":
        # 测试: 加载一只股票并打印
        code = (args.stocks or ["600519"])[0]
        df = load_stock(code)
        if df is not None:
            print(f"\n{code} 数据:")
            print(f"  行数: {len(df)}")
            print(f"  时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
            print(f"\n  前5行:")
            print(df.head().to_string())
            
            feat = prepare_features(code)
            if feat is not None:
                print(f"\n  特征维度: {feat.shape}")
        else:
            print(f"  {code} 无缓存，请先执行: python stock_data.py update")
