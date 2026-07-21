import { createBrowserClient } from "@supabase/ssr";
import { normalizeSupabaseUrl } from "./supabaseUrl";

// Anon-key client for "use client" components (login form, sign-out button).
// Safe to expose: relies on RLS + the anon key's limited privileges.
export function getSupabaseBrowserClient() {
  return createBrowserClient(
    normalizeSupabaseUrl(process.env.NEXT_PUBLIC_SUPABASE_URL!),
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
  );
}