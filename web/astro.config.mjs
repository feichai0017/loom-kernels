// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// QuillCache docs site — Astro + Starlight, Claude-themed.
// Deployed to GitHub Pages at https://feichai0017.github.io/quillcache/
export default defineConfig({
  site: "https://feichai0017.github.io",
  base: "/quillcache",
  integrations: [
    starlight({
      title: "QuillCache",
      description:
        "A Mooncake/Dynamo-style distributed KV cache pool and control plane for LLM serving, in Rust — with identity-governed safe reuse and a crash-consistent persistent tier.",
      social: {
        github: "https://github.com/feichai0017/quillcache",
      },
      customCss: ["./src/styles/claude.css"],
      head: [
        {
          tag: "link",
          attrs: { rel: "preconnect", href: "https://fonts.googleapis.com" },
        },
        {
          tag: "link",
          attrs: {
            rel: "preconnect",
            href: "https://fonts.gstatic.com",
            crossorigin: true,
          },
        },
        {
          tag: "link",
          attrs: {
            rel: "stylesheet",
            href: "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;450;500;600&display=swap",
          },
        },
      ],
      editLink: {
        baseUrl: "https://github.com/feichai0017/quillcache/edit/main/web/",
      },
      lastUpdated: true,
      sidebar: [
        {
          label: "Start here",
          items: [
            { label: "Overview", link: "/overview/" },
            { label: "Quick start", link: "/quickstart/" },
          ],
        },
        {
          label: "Architecture",
          items: [
            { label: "How it fits together", link: "/architecture/" },
            { label: "Crates", link: "/crates/" },
            { label: "Mooncake / Dynamo mapping", link: "/reference-mapping/" },
          ],
        },
        {
          label: "Deep dives",
          items: [
            { label: "ART vs LSM storage study", link: "/storage-study/" },
            { label: "Identity-safe reuse", link: "/identity-safe-reuse/" },
            { label: "Crash-consistent tier", link: "/crash-consistency/" },
          ],
        },
      ],
    }),
  ],
});
