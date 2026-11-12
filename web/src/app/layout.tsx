import type { Metadata } from "next";
import TopBar from "@/components/TopBar";
import { COPY } from "@/lib/copy";
import "./globals.css";

export const metadata: Metadata = {
  title: COPY.pageTitle,
  description: COPY.tagline,
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="es">
      <body>
        <TopBar />
        <main>{children}</main>
      </body>
    </html>
  );
}
