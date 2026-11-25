import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/ThemeProvider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "TensorCanvas — Build",
  description: "Visually build, inspect and intervene on neural networks.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  // Set the saved palette before paint to avoid a flash of the default.
  const paletteScript =
    "try{document.documentElement.dataset.palette=localStorage.getItem('tensorcanvas:palette')||'platinum'}catch(e){document.documentElement.dataset.palette='platinum'}";
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: paletteScript }} />
      </head>
      <body className="h-full">
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
