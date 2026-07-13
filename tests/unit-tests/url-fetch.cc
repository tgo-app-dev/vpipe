// url-fetch.cc -- tests for the web tools: the SSRF guard (deterministic,
// no live network needed -- private/localhost targets are refused before
// or at connect), the HTML-to-text reducer, and the tool wrappers. A live
// fetch against the public internet is gated on VPIPE_MCP_NET_TEST=1.

#include "minitest.h"

#include "common/flex-data.h"
#include "generative-models/shared/mcp/mcp-tools.h"
#include "generative-models/shared/mcp/url-fetch.h"

#include <cstdlib>
#include <string>

using namespace std;
using namespace vpipe;

// ---- SSRF guard (URL literals refused before any DNS) ---------------

TEST(url_fetch, blocks_non_http_scheme)
{
  auto r = fetch_url("file:///etc/passwd");
  EXPECT_FALSE(r.ok);
  EXPECT_TRUE(r.error.find("http") != string::npos);
}

TEST(url_fetch, blocks_localhost)
{
  auto r = fetch_url("http://localhost:8080/x");
  EXPECT_FALSE(r.ok);
  EXPECT_TRUE(r.error.find("blocked") != string::npos);
}

TEST(url_fetch, blocks_loopback_literal)
{
  auto r = fetch_url("http://127.0.0.1/x");
  EXPECT_FALSE(r.ok);
  EXPECT_TRUE(r.error.find("blocked") != string::npos);
}

TEST(url_fetch, blocks_private_literal)
{
  EXPECT_FALSE(fetch_url("http://10.0.0.5/").ok);
  EXPECT_FALSE(fetch_url("http://192.168.1.1/").ok);
  EXPECT_FALSE(fetch_url("http://172.16.0.1/").ok);
}

TEST(url_fetch, blocks_cloud_metadata)
{
  // The AWS/GCP metadata endpoint is link-local (169.254/16).
  auto r = fetch_url("http://169.254.169.254/latest/meta-data/");
  EXPECT_FALSE(r.ok);
  EXPECT_TRUE(r.error.find("blocked") != string::npos);
}

TEST(url_fetch, blocks_ipv6_loopback)
{
  auto r = fetch_url("http://[::1]:9000/");
  EXPECT_FALSE(r.ok);
  EXPECT_TRUE(r.error.find("blocked") != string::npos);
}

TEST(url_fetch, allow_private_opt_in_changes_gate)
{
  // With allow_private the loopback literal is no longer refused for the
  // SSRF reason; it fails later (connection refused), not "blocked".
  UrlFetchOptions o;
  o.allow_private = true;
  o.timeout_seconds = 2;
  auto r = fetch_url("http://127.0.0.1:1/", o);
  EXPECT_FALSE(r.ok);
  EXPECT_TRUE(r.error.find("blocked") == string::npos);
}

TEST(url_fetch, tool_blocks_and_reports_error)
{
  McpTool t = make_fetch_url_tool();
  const string out = t.handler("{\"url\":\"http://169.254.169.254/\"}");
  FlexData fd = FlexData::from_json(out);
  EXPECT_TRUE(fd.is_object());
  EXPECT_TRUE(fd.as_object().contains("error"));
}

TEST(url_fetch, tool_missing_url_reports_error)
{
  McpTool t = make_scrape_page_tool();
  FlexData fd = FlexData::from_json(t.handler("{}"));
  EXPECT_TRUE(fd.is_object() && fd.as_object().contains("error"));
}

// ---- HTML -> text ---------------------------------------------------

TEST(url_fetch, html_to_text_strips_tags_and_scripts)
{
  const string html =
      "<html><head><title>T</title>"
      "<style>.x{color:red}</style></head>"
      "<body><h1>Hello</h1>"
      "<script>alert('nope')</script>"
      "<p>World &amp; peace</p></body></html>";
  const string text = html_to_text(html, 4096);
  EXPECT_TRUE(text.find("Hello") != string::npos);
  EXPECT_TRUE(text.find("World & peace") != string::npos);
  // Script/style content must be gone.
  EXPECT_TRUE(text.find("alert") == string::npos);
  EXPECT_TRUE(text.find("color:red") == string::npos);
  // No angle brackets remain.
  EXPECT_TRUE(text.find('<') == string::npos);
}

TEST(url_fetch, html_to_text_decodes_entities)
{
  const string t = html_to_text(
      "<p>a&lt;b &#65; &#x42; &nbsp;end</p>", 4096);
  EXPECT_TRUE(t.find("a<b") != string::npos);
  EXPECT_TRUE(t.find("A") != string::npos);   // &#65;
  EXPECT_TRUE(t.find("B") != string::npos);   // &#x42;
}

TEST(url_fetch, html_to_text_caps_length)
{
  string big = "<p>";
  for (int i = 0; i < 1000; ++i) { big += "word "; }
  big += "</p>";
  const string t = html_to_text(big, 100);
  EXPECT_TRUE(t.size() <= 100);
}

TEST(url_fetch, html_title_extracted)
{
  EXPECT_TRUE(html_title("<html><title>Hi There</title></html>")
              == "Hi There");
  EXPECT_TRUE(html_title("<html>no title</html>").empty());
}

// ---- Live fetch (opt-in; needs public internet) ---------------------

TEST(url_fetch, live_example_com)
{
  const char* on = std::getenv("VPIPE_MCP_NET_TEST");
  if (!on || !*on) { return; }
  auto r = fetch_url("https://example.com/");
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.status == 200);
  EXPECT_TRUE(r.body.find("Example Domain") != string::npos);
  const string text = html_to_text(r.body, 8192);
  EXPECT_TRUE(text.find("Example Domain") != string::npos);
}
