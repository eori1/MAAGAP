import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MAAGAP – Predictive Intelligence for Smarter Provincial Governance",
  description:
    "MAAGAP is an AI-driven governance platform that helps government agencies monitor, predict, and optimize public projects.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" style={{ colorScheme: "light", background: "#ffffff" }}>
      <body style={{ background: "#ffffff", margin: 0, padding: 0, height: "100%" }}>
        {children}
      </body>
    </html>
  );
}
