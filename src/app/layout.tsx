import type { Metadata } from "next";
import "./globals.css";
import { ANGUSHT_VERSION_SHORT } from "@/lib/version";

export const metadata: Metadata = {
  title: `Angusht ${ANGUSHT_VERSION_SHORT} — Нейроморфная когнитивная архитектура`,
  description: "6 ядер x 216K LIF-нейронов с STDP, веб-поиском и самообучением.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru" suppressHydrationWarning>
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
