import ccxt


class ExchangeClient:
    def __init__(self, rate_limit=True):
        # futures client (binanceusdm) and spot (binance) as fallback
        self.futures = None
        self.spot = None
        try:
            if hasattr(ccxt, 'binanceusdm'):
                self.futures = ccxt.binanceusdm({"enableRateLimit": rate_limit})
            else:
                # unified binance class can be used for futures by setting options
                self.futures = ccxt.binance({"enableRateLimit": rate_limit})
                self.futures.options.setdefault('defaultType', 'future')
            try:
                self.futures.load_markets()
            except Exception:
                pass
        except Exception:
            self.futures = None

        try:
            self.spot = ccxt.binance({"enableRateLimit": rate_limit})
            try:
                self.spot.load_markets()
            except Exception:
                pass
        except Exception:
            self.spot = None

    def fetch_ohlcv(self, symbol, timeframe="15m", limit=500):
        """Try to fetch OHLCV from futures first, then fallback to spot.

        symbol should be like 'BTC/USDT' (no ':USDT' suffix).
        Tries multiple symbol formats for better coverage.
        Raises the last exception if all attempts fail.
        """
        # Generate multiple symbol variants to try
        base_symbol = symbol.replace(':USDT', '')  # BTC/USDT

        symbol_variants = [
            base_symbol,                    # BTC/USDT
            f"{base_symbol}:USDT",         # BTC/USDT:USDT
            symbol,                         # original input
        ]

        # Add 1000-prefixed variants for meme coins that might use different prefixes
        if '/' in base_symbol:
            token = base_symbol.split('/')[0]
            if not token.startswith('1000'):
                symbol_variants.append(f"1000{token}/USDT")
                symbol_variants.append(f"1000{token}/USDT:USDT")

        last_exc = None

        # try futures first with all variants
        if self.futures:
            for variant in symbol_variants:
                try:
                    markets = getattr(self.futures, 'markets', {})
                    if variant in markets:
                        return self.futures.fetch_ohlcv(variant, timeframe=timeframe, limit=limit)
                except Exception as e:
                    last_exc = e

            # try naive fetch without market check
            for variant in symbol_variants:
                try:
                    return self.futures.fetch_ohlcv(variant, timeframe=timeframe, limit=limit)
                except Exception as e:
                    last_exc = e

        # try spot with all variants
        if self.spot:
            for variant in symbol_variants:
                try:
                    markets = getattr(self.spot, 'markets', {})
                    if variant in markets:
                        return self.spot.fetch_ohlcv(variant, timeframe=timeframe, limit=limit)
                except Exception as e:
                    last_exc = e

            # try naive fetch without market check
            for variant in symbol_variants:
                try:
                    return self.spot.fetch_ohlcv(variant, timeframe=timeframe, limit=limit)
                except Exception as e:
                    last_exc = e

        if last_exc:
            raise last_exc
        raise RuntimeError(f"could not fetch ohlcv for {symbol} (tried variants: {symbol_variants})")
