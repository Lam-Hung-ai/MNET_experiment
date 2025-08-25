// app/layout.tsx
import FooterSection from "@/components/footer";
import "./globals.css";
import type { Metadata } from "next";
import { HeroHeader } from "@/components/header";

export const metadata: Metadata = {
  title: "MNet",
  description: "model MNet với khả năng xóa watermark hiệu quả",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <HeroHeader />
        <main className="bg-gradient-to-bl from-pink-50 via-rose-100 to-rose-200 pt-20 lg:pt-24 pb-20 min-h-screen">
          {children}
        </main>
        <FooterSection />
      </body>
    </html>
  )
}

