#pragma once
#include "types.hpp"
#include <string>

// Fetch the Binance Futures depth snapshot synchronously via HTTPS.
// Throws std::runtime_error on network or parse failure.
OrderBookSnapshot fetch_depth_snapshot(const std::string& symbol = "BTCUSDT",
                                       int                 limit  = 1000);
