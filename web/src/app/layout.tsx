import type { Metadata } from "next";
import TopBar from "@/components/TopBar";
import { ChatProvider } from "@/lib/chat-context";
import { COPY } from "@/lib/copy";
import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";

export const metadata: Metadata = {
  title: COPY.pageTitle,
  description: COPY.tagline,
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="es">
      <body>
        <ChatProvider>
          <TopBar />
          <main>{children}</main>
        </ChatProvider>
      </body>
    </html>
  );
}
