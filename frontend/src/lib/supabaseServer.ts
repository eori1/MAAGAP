import { createClient } from "@supabase/supabase-js";

// Server-only client (used inside Route Handlers, never imported into
// client components). Uses the service-role key so it must never be
// referenced from "use client" code or prefixed with NEXT_PUBLIC_.
let cached: ReturnType<typeof createClient> | null = null;

export function getSupabaseServerClient() {
  if (cached) return cached;

  let url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) {
    throw new Error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set in frontend/.env.local");
  }

  // The client appends /rest/v1 itself; normalize in case the project URL
  // was copied with a REST path already attached (see backend/maagap/database.py).
  url = url.replace(/\/$/, "");
  for (const suffix of ["/rest/v1", "/rest"]) {
    if (url.endsWith(suffix)) {
      url = url.slice(0, -suffix.length);
      break;
    }
  }

  cached = createClient(url, key, {
    auth: { persistSession: false },
  });
  return cached;
}
