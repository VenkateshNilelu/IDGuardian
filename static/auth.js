// IDGuardian login/signup page logic (Supabase Auth: Google OAuth + email/password).
//
// window.SUPABASE_URL / window.SUPABASE_ANON_KEY come from login.html, which
// gets them from app.py's SUPABASE_URL / SUPABASE_ANON_KEY env vars. The
// anon key is meant to be used client-side by design (Supabase enforces
// access via Row Level Security, not by keeping this key secret) -- it is
// not the same thing as a service-role key, which must never appear here.

let mode = "signin"; // "signin" | "signup"
let supabaseClient = null;

function initSupabase() {
  // NOTE: the CDN script below declares its own global `supabase` (the SDK
  // namespace, with .createClient()) -- this file's own client is deliberately
  // named `supabaseClient` to avoid a global `let`/`var` redeclaration clash
  // with it (that clash threw an uncaught SyntaxError that broke this whole
  // page, caught only once real Supabase credentials made this path run).
  if (!window.SUPABASE_URL || !window.SUPABASE_ANON_KEY) return null;
  if (!window.supabase || !window.supabase.createClient) return null;
  return window.supabase.createClient(window.SUPABASE_URL, window.SUPABASE_ANON_KEY);
}

function showMessage(text, kind) {
  const el = document.getElementById("authMsg");
  el.textContent = text;
  el.className = "text-sm mt-4 rounded-lg px-3 py-2 " + (
    kind === "error" ? "bg-red-50 text-red-700 border border-red-200" :
    kind === "success" ? "bg-emerald-50 text-emerald-700 border border-emerald-200" :
    "bg-blue-50 text-blue-700 border border-blue-200"
  );
  el.hidden = false;
}

function setMode(next) {
  mode = next;
  const isSignIn = mode === "signin";
  document.getElementById("tabSignIn").classList.toggle("bg-white", isSignIn);
  document.getElementById("tabSignIn").classList.toggle("shadow-sm", isSignIn);
  document.getElementById("tabSignIn").classList.toggle("text-slate-900", isSignIn);
  document.getElementById("tabSignIn").classList.toggle("text-slate-500", !isSignIn);
  document.getElementById("tabSignIn").setAttribute("aria-pressed", String(isSignIn));

  document.getElementById("tabSignUp").classList.toggle("bg-white", !isSignIn);
  document.getElementById("tabSignUp").classList.toggle("shadow-sm", !isSignIn);
  document.getElementById("tabSignUp").classList.toggle("text-slate-900", !isSignIn);
  document.getElementById("tabSignUp").classList.toggle("text-slate-500", isSignIn);
  document.getElementById("tabSignUp").setAttribute("aria-pressed", String(!isSignIn));

  document.getElementById("formTitle").textContent = isSignIn ? "Welcome back" : "Create your account";
  document.getElementById("formSub").textContent = isSignIn
    ? "Sign in to sync your analysis history across devices."
    : "Sign up to keep your analysis history and saved reports across devices.";
  document.getElementById("submitBtn").textContent = isSignIn ? "Sign in" : "Create account";
  document.getElementById("authMsg").hidden = true;
}

async function handleEmailSubmit(ev) {
  ev.preventDefault();
  if (!supabaseClient) {
    showMessage("Sign-in isn't configured yet — SUPABASE_URL/SUPABASE_ANON_KEY aren't set.", "error");
    return;
  }
  const email = document.getElementById("emailInput").value.trim();
  const password = document.getElementById("passwordInput").value;
  const btn = document.getElementById("submitBtn");
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Please wait…";

  try {
    if (mode === "signup") {
      const { error } = await supabaseClient.auth.signUp({ email, password });
      if (error) throw error;
      showMessage("Account created — check your email to confirm, then sign in.", "success");
      setMode("signin");
    } else {
      const { error } = await supabaseClient.auth.signInWithPassword({ email, password });
      if (error) throw error;
      window.location.href = "/dashboard";
    }
  } catch (err) {
    showMessage(err.message || "Something went wrong.", "error");
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}

async function handleGoogleClick() {
  if (!supabaseClient) {
    showMessage("Sign-in isn't configured yet — SUPABASE_URL/SUPABASE_ANON_KEY aren't set.", "error");
    return;
  }
  const { error } = await supabaseClient.auth.signInWithOAuth({
    provider: "google",
    options: { redirectTo: window.location.origin + "/dashboard" },
  });
  if (error) showMessage(error.message, "error");
  // On success, Supabase redirects the browser to Google then back to redirectTo --
  // nothing else to do here.
}

async function redirectIfAlreadySignedIn() {
  if (!supabaseClient) return;
  const { data } = await supabaseClient.auth.getSession();
  if (data && data.session) window.location.href = "/dashboard";
}

document.addEventListener("DOMContentLoaded", () => {
  supabaseClient = initSupabase();
  document.getElementById("tabSignIn").addEventListener("click", () => setMode("signin"));
  document.getElementById("tabSignUp").addEventListener("click", () => setMode("signup"));
  document.getElementById("authForm").addEventListener("submit", handleEmailSubmit);
  document.getElementById("googleBtn").addEventListener("click", handleGoogleClick);
  redirectIfAlreadySignedIn();
});
