import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// jsdom does not implement window scrolling; the chat follows the page bottom with it.
if (typeof window !== "undefined") window.scrollTo = () => {};

afterEach(() => cleanup());
