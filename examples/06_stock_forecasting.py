"""
Example 6: Stock/ETF Price Forecasting with Real Market Data

This example demonstrates how to:
1. Fetch real-time ETF/stock data using akshare
2. Preprocess financial time series data
3. Use forecasting models to predict future prices
4. Visualize predictions and evaluate performance

Requirements:
    pip install akshare openpyxl

Usage:
    python examples/06_stock_forecasting.py
    python examples/06_stock_forecasting.py --etf-code 159565 --days 365
    python examples/06_stock_forecasting.py --model chronos2
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import from installed package first, then fallback to local src
try:
    from time_series_forecasting_zero.models.unified import UnifiedForecaster
    from time_series_forecasting_zero.utils.visualizer import ForecastVisualizer
    from time_series_forecasting_zero.utils.forecast_utils import compute_all_metrics, print_metrics
except ImportError:
    from src.time_series_forecasting_zero.models.unified import UnifiedForecaster
    from src.time_series_forecasting_zero.utils.visualizer import ForecastVisualizer
    from src.time_series_forecasting_zero.utils.forecast_utils import compute_all_metrics, print_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stock/ETF Price Forecasting Example")
    parser.add_argument("--etf-code", type=str, default="159565",
                       help="ETF or stock code (default: 159565)")
    parser.add_argument("--days", type=int, default=365,
                       help="Number of days of historical data (default: 365)")
    parser.add_argument("--model", type=str, default="chronos2",
                       choices=["chronos2", "timesfm", "tirex"],
                       help="Model to use for forecasting (default: chronos2)")
    parser.add_argument("--forecast-days", type=int, default=None,
                       help="Number of days to forecast (default: 20%% of historical data)")
    return parser.parse_args()


def fetch_etf_data(etf_code: str, start_date: str, end_date: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch ETF daily price data using akshare with retry mechanism.
    
    Args:
        etf_code: ETF code (e.g., "159565" for 电网ETF)
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        max_retries: Maximum number of retry attempts (default: 3)
    
    Returns:
        DataFrame with columns: date, open, close, high, low, volume
    """
    try:
        import akshare as ak
        import time
        
        print(f"[Step 1] Fetching ETF data from akshare...")
        print(f"  ETF Code: {etf_code}")
        print(f"  Date Range: {start_date} to {end_date}")
        
        # Try fetching data with retries
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  Attempt {attempt}/{max_retries}...")
                
                # Fetch ETF historical data from East Money
                etf_df = ak.fund_etf_hist_em(
                    symbol=etf_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date
                )
                
                if etf_df is None or etf_df.empty:
                    raise ValueError("No data returned from akshare")
                
                print(f"  [OK] Successfully fetched {len(etf_df)} trading days")
                print(f"  Columns available: {list(etf_df.columns)}")
                
                return etf_df
                
            except Exception as e:
                last_error = e
                print(f"  [WARNING] Attempt {attempt} failed: {type(e).__name__}: {str(e)[:100]}")
                
                if attempt < max_retries:
                    wait_time = 2 * attempt  # Exponential backoff: 2s, 4s, 6s
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All retries failed
        print(f"\n  [ERROR] Failed to fetch data after {max_retries} attempts")
        print(f"  Last error: {last_error}")
        print(f"\n  Possible causes:")
        print(f"    1. Network connection issues")
        print(f"    2. API rate limiting from akshare/East Money")
        print(f"    3. Invalid ETF code: {etf_code}")
        print(f"    4. Temporary server unavailability")
        print(f"\n  Solutions:")
        print(f"    - Check your internet connection")
        print(f"    - Verify the ETF code is valid (e.g., 159565, 510300)")
        print(f"    - Try again later")
        print(f"    - Use a different model example (e.g., examples/01_basic_forecasting.py)")
        
        # Offer to use demo data
        print(f"\n  Would you like to use demo data instead? (y/n): ", end="")
        try:
            choice = input().strip().lower()
            if choice == 'y':
                print(f"  Generating demo data for demonstration...")
                return generate_demo_etf_data(etf_code, start_date, end_date)
        except:
            pass
        
        sys.exit(1)
        
    except ImportError:
        print("  [ERROR] akshare not installed. Please run: pip install akshare")
        sys.exit(1)


def generate_demo_etf_data(etf_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate demo ETF data for testing when API is unavailable.
    
    Args:
        etf_code: ETF code for labeling
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
    
    Returns:
        DataFrame with simulated ETF data
    """
    print(f"  [INFO] Generating demo data for ETF {etf_code}")
    
    # Create date range (excluding weekends)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    
    # Simulate realistic ETF price movement using random walk
    np.random.seed(42)  # For reproducibility
    initial_price = 1.0  # Starting price
    
    # Generate price series with trend and volatility
    daily_returns = np.random.normal(0.0005, 0.015, n_days)  # Mean return 0.05%, vol 1.5%
    prices = initial_price * np.cumprod(1 + daily_returns)
    
    # Add some realistic features
    open_prices = prices * (1 + np.random.uniform(-0.005, 0.005, n_days))
    high_prices = np.maximum(open_prices, prices) * (1 + np.abs(np.random.normal(0, 0.008, n_days)))
    low_prices = np.minimum(open_prices, prices) * (1 - np.abs(np.random.normal(0, 0.008, n_days)))
    volumes = np.random.randint(1000000, 10000000, n_days)  # Random volumes
    
    # Create DataFrame with Chinese column names (as akshare returns)
    df = pd.DataFrame({
        '日期': dates.strftime('%Y-%m-%d'),
        '开盘': np.round(open_prices, 4),
        '收盘': np.round(prices, 4),
        '最高': np.round(high_prices, 4),
        '最低': np.round(low_prices, 4),
        '成交量': volumes
    })
    
    print(f"  [OK] Generated {len(df)} days of demo data")
    print(f"  Price range: ¥{prices.min():.4f} - ¥{prices.max():.4f}")
    
    return df


def preprocess_financial_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess financial time series data for forecasting.
    
    Args:
        raw_df: Raw DataFrame from akshare
    
    Returns:
        Processed DataFrame with datetime index and 'close' column
    """
    print("\n[Step 2] Preprocessing financial data...")
    
    # Rename Chinese columns to English if needed
    column_mapping = {
        '日期': 'date',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'turnover'
    }
    
    df = raw_df.rename(columns={k: v for k, v in column_mapping.items() if k in raw_df.columns})
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
    
    # Ensure numeric types
    numeric_cols = ['open', 'close', 'high', 'low', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN values
    df.dropna(subset=['close'], inplace=True)
    
    print(f"  [OK] Processed data shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Close price range: ¥{df['close'].min():.2f} - ¥{df['close'].max():.2f}")
    
    return df


def prepare_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split data into training and testing sets.
    
    Args:
        df: DataFrame with 'close' column
        test_ratio: Ratio of data to use for testing
    
    Returns:
        train_data, test_data, train_dates, test_dates
    """
    print("\n[Step 3] Preparing train/test split...")
    
    close_prices = df['close'].values
    dates = df.index
    
    # Calculate split point
    train_size = int(len(close_prices) * (1 - test_ratio))
    
    train_data = close_prices[:train_size]
    test_data = close_prices[train_size:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]
    
    print(f"  Training samples: {len(train_data)} ({train_dates[0]} to {train_dates[-1]})")
    print(f"  Testing samples: {len(test_data)} ({test_dates[0]} to {test_dates[-1]})")
    
    return train_data, test_data, train_dates, test_dates


def forecast_with_model(model_name: str, train_data: np.ndarray, 
                       forecast_horizon: int, device: str) -> dict:
    """
    Make predictions using specified model.
    
    Args:
        model_name: Model name (chronos2, timesfm, tirex)
        train_data: Training data array
        forecast_horizon: Number of steps to forecast
        device: Device to run model on
    
    Returns:
        Dictionary with predictions
    """
    print(f"\n[Step 4] Loading {model_name.upper()} model...")
    
    forecaster = UnifiedForecaster(
        model_name=model_name,
        forecast_horizon=forecast_horizon,
        device=device
    )
    
    print(f"  Loading model from local storage...")
    forecaster.load_model()
    
    print(f"  Making predictions for {forecast_horizon} days...")
    predictions = forecaster.predict(
        context=train_data,
        forecast_horizon=forecast_horizon,
        quantiles=[0.1, 0.5, 0.9]
    )
    
    print(f"  [OK] Prediction completed")
    print(f"  Predicted price range: ¥{predictions['mean'].min():.2f} - ¥{predictions['mean'].max():.2f}")
    
    return predictions


def visualize_results(train_data: np.ndarray, test_data: np.ndarray,
                     train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex,
                     predictions: dict, model_name: str, output_dir: Path):
    """
    Create visualization of forecasting results.
    
    Args:
        train_data: Training data
        test_data: Test data
        train_dates: Training dates
        test_dates: Test dates
        predictions: Prediction results
        model_name: Model name used
        output_dir: Directory to save plots
    """
    print("\n[Step 5] Creating visualizations...")
    
    visualizer = ForecastVisualizer()
    
    # Combine all data for plotting
    all_dates = np.concatenate([train_dates, test_dates])
    all_actual = np.concatenate([train_data, test_data])
    
    # Convert to pandas Series for better plotting
    hist_series = pd.Series(train_data, index=train_dates)
    
    # Plot forecast with uncertainty
    fig_path = output_dir / f"{model_name}_forecast.png"
    fig = visualizer.plot_forecast(
        historical=hist_series,
        predictions=predictions,
        timestamps_hist=train_dates,
        timestamps_forecast=test_dates,
        title=f'{model_name.upper()} - ETF Price Forecast',
        ylabel='Price (¥)',
        save_path=str(fig_path)
    )
    
    print(f"  [OK] Forecast plot saved: {fig_path}")
    
    # Plot residuals
    residual_path = output_dir / f"{model_name}_residuals.png"
    fig_res = visualizer.plot_residuals(
        y_true=test_data,
        y_pred=predictions['mean'],
        title=f'{model_name.upper()} - Forecast Residuals',
        save_path=str(residual_path)
    )
    
    print(f"  [OK] Residual plot saved: {residual_path}")


def main():
    """Run stock/ETF forecasting example."""
    args = parse_args()
    
    print("="*70)
    print("Stock/ETF Price Forecasting Example")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  ETF/Stock Code: {args.etf_code}")
    print(f"  Historical Days: {args.days}")
    print(f"  Forecasting Model: {args.model.upper()}")
    
    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Calculate date range
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y%m%d")
    
    # Step 1: Fetch data
    raw_df = fetch_etf_data(args.etf_code, start_date, end_date)
    
    # Step 2: Preprocess data
    df = preprocess_financial_data(raw_df)
    
    # Display core data
    print("\n【Core Price Data】")
    core_data = df[['open', 'close', 'high', 'low', 'volume']].copy()
    core_data.index = core_data.index.strftime('%Y-%m-%d')
    print(core_data.tail(10))  # Show last 10 days
    
    # Step 3: Prepare train/test split
    forecast_days = args.forecast_days if args.forecast_days else int(len(df) * 0.2)
    test_ratio = forecast_days / len(df)
    train_data, test_data, train_dates, test_dates = prepare_train_test_split(
        df, test_ratio=test_ratio
    )
    
    # Step 4: Make predictions
    predictions = forecast_with_model(
        model_name=args.model,
        train_data=train_data,
        forecast_horizon=len(test_data),
        device=device
    )
    
    # Step 5: Evaluate metrics
    print("\n[Step 6] Evaluating forecast accuracy...")
    metrics = compute_all_metrics(
        y_true=test_data,
        y_pred=predictions['mean'],
        lower_bound=predictions.get('lower_bound'),
        upper_bound=predictions.get('upper_bound'),
        y_train=train_data
    )
    
    print_metrics(metrics, title=f"{args.model.upper()} Forecast Performance")
    
    # Additional financial metrics
    print("\n【Financial Metrics】")
    print(f"  Mean Absolute Error: ¥{metrics['mae']:.2f}")
    print(f"  Root Mean Squared Error: ¥{metrics['rmse']:.2f}")
    print(f"  Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
    
    # Direction accuracy (did we predict the right direction?)
    if len(test_data) > 1:
        actual_direction = np.diff(test_data) > 0
        pred_direction = np.diff(predictions['mean']) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        print(f"  Direction Accuracy: {direction_accuracy:.1f}%")
    
    # Step 6: Visualize results
    output_dir = project_root / "outputs" / "stock_forecasting"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_results(
        train_data=train_data,
        test_data=test_data,
        train_dates=train_dates,
        test_dates=test_dates,
        predictions=predictions,
        model_name=args.model,
        output_dir=output_dir
    )
    
    # Step 7: Save predictions to CSV
    print("\n[Step 7] Saving predictions to CSV...")
    pred_df = pd.DataFrame({
        'date': test_dates,
        'actual_price': test_data,
        'predicted_price': predictions['mean'],
        'lower_bound': predictions['lower_bound'],
        'upper_bound': predictions['upper_bound']
    })
    
    csv_path = output_dir / f"{args.etf_code}_{args.model}_predictions.csv"
    pred_df.to_csv(csv_path, index=False)
    print(f"  [OK] Predictions saved to: {csv_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("Forecasting Summary")
    print("="*70)
    print(f"  ETF Code: {args.etf_code}")
    print(f"  Model: {args.model.upper()}")
    print(f"  Training Period: {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Testing Period: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  RMSE: ¥{metrics['rmse']:.2f}")
    print(f"  MAE: ¥{metrics['mae']:.2f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print("="*70)
    print("\nExample completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
