// Gates the dashboard (templates/index.html) behind a signed-in Supabase
// session. Landing-page CTAs link straight to /dashboard, so without this
// an anonymous visitor could reach the product without ever going through
// /login -- a bookmark, a shared link, or just typing the URL would all
// skip signup too. This is what actually enforces "landing page -> sign up
// -> product", by gating the one real entry point instead of chasing every
// place a link to /dashboard could come from.
//
// Runs synchronously from <head>, before <body> paints, so an unauthenticated
// visitor never sees a flash of the real dashboard before being redirected.
// Client-side only (matches this app's auth architecture end to end -- no
// server-side session), so it's a UX gate, not an access-control boundary.
(function () {
  if (!window.SUPABASE_URL || !window.SUPABASE_ANON_KEY) return; // not configured -- no auth system to enforce
  if (!window.supabase || !window.supabase.createClient) return;

  document.documentElement.style.visibility = "hidden";

  var client = window.supabase.createClient(window.SUPABASE_URL, window.SUPABASE_ANON_KEY);
  var decided = false;
  client.auth.onAuthStateChange(function (_event, session) {
    if (decided) return; // only the first firing decides this page load; later sign-outs are handled by the sign-out button's own redirect
    decided = true;
    if (session) {
      document.documentElement.style.visibility = "";
    } else {
      window.location.replace("/login");
    }
  });
})();
