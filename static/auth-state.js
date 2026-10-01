// Navbar auth-state widget for the main dashboard (templates/index.html):
// shows a "Sign in" link when logged out, or an avatar + dropdown (email,
// Sign out) when logged in. Separate from app.js since it's a small,
// self-contained concern and app.js is already large.
//
// Scope note: this only reflects sign-in state in the navbar. History/Saved
// Reports still key off the anonymous localStorage client_id (see app.js) --
// tying them to the signed-in Supabase user instead is a natural follow-up,
// not done here.

(function () {
  function getClient() {
    if (!window.SUPABASE_URL || !window.SUPABASE_ANON_KEY) return null;
    if (!window.supabase || !window.supabase.createClient) return null;
    return window.supabase.createClient(window.SUPABASE_URL, window.SUPABASE_ANON_KEY);
  }

  function showSignedOut() {
    document.getElementById("signInLink").hidden = false;
    document.getElementById("userMenuBtn").hidden = true;
    document.getElementById("userMenu").hidden = true;
  }

  function showSignedIn(user) {
    const btn = document.getElementById("userMenuBtn");
    const initial = (user.email || "?").charAt(0).toUpperCase();
    btn.textContent = initial;
    btn.hidden = false;
    document.getElementById("signInLink").hidden = true;
    document.getElementById("userMenuEmail").textContent = user.email || "";
  }

  document.addEventListener("DOMContentLoaded", () => {
    const supabase = getClient();
    if (!supabase) { showSignedOut(); return; }

    // Wiring onAuthStateChange alone (not also a one-off getSession() call)
    // is deliberate: when a page loads straight off an OAuth redirect,
    // Supabase is still asynchronously exchanging the ?code= param in the
    // URL for a session. onAuthStateChange fires once immediately with
    // whatever the current state is AND again when that exchange finishes
    // -- a separate getSession() call races that exchange and can resolve
    // first, showing "Sign in" even though the user is, a moment later,
    // actually signed in.
    supabase.auth.onAuthStateChange((_event, session) => {
      if (session) showSignedIn(session.user); else showSignedOut();
    });

    document.getElementById("userMenuBtn").addEventListener("click", () => {
      const menu = document.getElementById("userMenu");
      menu.hidden = !menu.hidden;
    });
    document.addEventListener("click", (ev) => {
      const authState = document.getElementById("authState");
      if (!authState.contains(ev.target)) document.getElementById("userMenu").hidden = true;
    });
    document.getElementById("signOutBtn").addEventListener("click", async () => {
      await supabase.auth.signOut();
      window.location.href = "/";
    });
  });
})();
