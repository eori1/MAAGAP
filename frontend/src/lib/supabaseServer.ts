import { createClient } from "@supabase/supabase-js";
import { normalizeSupabaseUrl } from "./supabaseUrl";

// Server-only client (used inside Route Handlers, never imported into
// client components). Uses the service-role key so it must never be
// referenced from "use client" code or prefixed with NEXT_PUBLIC_.
let cached: ReturnType<typeof createClient> | null = null;

export function getSupabaseServerClient() {
  if (cached) return cached;

  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) {
    throw new Error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set in frontend/.env.local");
  }

  cached = createClient(normalizeSupabaseUrl(url), key, {
    auth: { persistSession: false },
  });
  return cached;
}
