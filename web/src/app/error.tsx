"use client";

// Catches errors thrown by the (app) layout itself (e.g. the API is down or misconfigured),
// which the (app)/error.tsx boundary cannot: error.js does not wrap its own segment's layout.
export { default } from "./(app)/error";
