"""
Regression tests for parse_profile_input() -- specifically the dotted-username
bug (e.g. "chandranindira.co" was misread as a bare domain like "instagram.com"
and swallowed into an empty path/empty username, see app.py's docstring on
parse_profile_input for the full story).
"""
import pytest


@pytest.mark.parametrize("raw,expected", [
    # Bare usernames -- including ones containing a dot, which must NEVER be
    # misread as a domain. This is the regression this file exists to guard.
    ("chandranindira.co", (None, "chandranindira.co")),
    ("first.last", (None, "first.last")),
    ("qbuch_vip", (None, "qbuch_vip")),
    ("promo_aaravchakraborty30563", (None, "promo_aaravchakraborty30563")),
    # Full URLs with scheme
    ("https://www.instagram.com/someuser/", ("instagram", "someuser")),
    ("https://x.com/someuser", ("twitter", "someuser")),
    # Bare domain + path, no scheme
    ("instagram.com/someuser", ("instagram", "someuser")),
    ("x.com/someuser", ("twitter", "someuser")),
    ("facebook.com/someuser", ("facebook", "someuser")),
    # LinkedIn /in/ and /company/ prefix stripping
    ("linkedin.com/in/someuser", ("linkedin", "someuser")),
    ("linkedin.com/company/someorg", ("linkedin", "someorg")),
])
def test_parse_profile_input(app_module, raw, expected):
    assert app_module.parse_profile_input(raw) == expected
