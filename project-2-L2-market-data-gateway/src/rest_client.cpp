#include "rest_client.hpp"
#include "parse_utils.hpp"

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <simdjson.h>

#include <stdexcept>
#include <string>
#include <iostream>
#include <charconv>
#include <algorithm>
#include <cctype>

namespace net   = boost::asio;
namespace ssl   = net::ssl;
namespace beast = boost::beast;
namespace http  = beast::http;
using     tcp   = net::ip::tcp;

// Bybit V5 unified market endpoint
static const char* HOST = "api.bybit.com";
static const char* PORT = "443";
static const char* PATH = "/v5/market/orderbook";

OrderBookSnapshot fetch_depth_snapshot(const std::string& symbol, int limit) {
    // Bybit accepts symbol in uppercase; normalise here.
    std::string upper = symbol;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    // Bybit V5 caps orderbook depth at 200 for linear futures.
    int bybit_limit = std::min(limit, 200);

    std::string target = std::string(PATH) +
                         "?category=linear" +
                         "&symbol="   + upper +
                         "&limit="    + std::to_string(bybit_limit);

    net::io_context ioc;
    ssl::context ctx{ssl::context::tlsv13_client};
    ctx.set_default_verify_paths();
    ctx.set_verify_mode(ssl::verify_peer);
    ctx.set_verify_callback(ssl::host_name_verification(HOST));

    beast::ssl_stream<beast::tcp_stream> stream{ioc, ctx};

    if (!SSL_set_tlsext_host_name(stream.native_handle(), HOST))
        throw std::runtime_error("SSL_set_tlsext_host_name failed");

    tcp::resolver resolver{ioc};
    auto results = resolver.resolve(HOST, PORT);
    beast::get_lowest_layer(stream).connect(results);
    stream.handshake(ssl::stream_base::client);

    http::request<http::empty_body> req{http::verb::get, target, 11};
    req.set(http::field::host, HOST);
    req.set(http::field::user_agent, "quant-day1/1.0");
    req.set(http::field::accept, "application/json");
    http::write(stream, req);

    beast::flat_buffer buf;
    http::response<http::string_body> res;
    http::read(stream, buf, res);

    if (res.result() != http::status::ok) {
        throw std::runtime_error("REST /orderbook returned HTTP " +
                                 std::to_string(static_cast<int>(res.result())));
    }

    // Parse JSON
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    auto err = parser.parse(res.body()).get(doc);
    if (err) throw std::runtime_error("simdjson parse error: " +
                                       std::string(simdjson::error_message(err)));

    OrderBookSnapshot snap;

    // === Bybit envelope check ===
    int64_t ret_code = -1;
    auto retcode_err = doc["retCode"].get_int64().get(ret_code);
    if (retcode_err != simdjson::SUCCESS) {
        throw std::runtime_error("FAIL at retCode: " +
                                 std::string(simdjson::error_message(retcode_err)));
    }
    if (ret_code != 0) {
        std::string_view ret_msg;
        (void) doc["retMsg"].get_string().get(ret_msg);
        throw std::runtime_error("Bybit retCode=" + std::to_string(ret_code) +
                                 " msg=" + std::string(ret_msg));
    }

    // === Extract result object ===
    simdjson::dom::element result_obj;
    auto result_err = doc["result"].get(result_obj);
    if (result_err != simdjson::SUCCESS) {
        throw std::runtime_error("FAIL at result: " +
                                 std::string(simdjson::error_message(result_err)));
    }

    // === Extract u with int64-then-uint64 fallback ===
    int64_t last_id_signed = 0;
    auto u_int_err = result_obj["u"].get_int64().get(last_id_signed);
    if (u_int_err != simdjson::SUCCESS) {
        uint64_t last_id_unsigned = 0;
        auto u_uint_err = result_obj["u"].get_uint64().get(last_id_unsigned);
        if (u_uint_err != simdjson::SUCCESS) {
            std::cerr << "[rest] FULL response body:\n" << res.body() << "\n";
            throw std::runtime_error("FAIL at result.u: int64=" +
                                     std::string(simdjson::error_message(u_int_err)) +
                                     " uint64=" + std::string(simdjson::error_message(u_uint_err)));
        }
        snap.last_update_id = last_id_unsigned;
    } else {
        snap.last_update_id = static_cast<uint64_t>(last_id_signed);
    }

    // === Parse bids and asks from result.b and result.a ===
    auto parse_levels = [](simdjson::dom::array arr) {
        std::vector<PriceLevel> out;
        out.reserve(arr.size());
        for (auto row : arr) {
            PriceLevel lv;
            std::string_view price_sv = row.at(0).get_string().value();
            std::string_view qty_sv   = row.at(1).get_string().value();

            if (!parse_scaled(price_sv, lv.price, PRICE_SCALE)) continue;
            if (!parse_scaled(qty_sv,   lv.qty,   QTY_SCALE))   continue;
            out.push_back(lv);
        }
        return out;
    };

    snap.bids = parse_levels(result_obj["b"].get_array());
    snap.asks = parse_levels(result_obj["a"].get_array());

    beast::error_code ec;
    stream.shutdown(ec);

    std::cout << "[rest] Bybit snapshot u=" << snap.last_update_id
              << "  bids=" << snap.bids.size()
              << "  asks=" << snap.asks.size() << "\n";
    return snap;
}