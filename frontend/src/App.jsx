import Lenis from "lenis";
import { useEffect } from "react";
import Bench from "./components/Bench";
import { Api, Footer, Nav } from "./components/Chrome";
import ContactSheet from "./components/ContactSheet";
import Hero from "./components/Hero";
import LightTable from "./components/LightTable";
import Method from "./components/Method";

export default function App() {
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return undefined;
    const lenis = new Lenis({ lerp: 0.09, wheelMultiplier: 0.95, anchors: { offset: 0 } });
    let id = requestAnimationFrame(function raf(t) {
      lenis.raf(t);
      id = requestAnimationFrame(raf);
    });
    return () => { cancelAnimationFrame(id); lenis.destroy(); };
  }, []);

  return (
    <>
      <div className="grain" aria-hidden="true" />
      <div className="vignette" aria-hidden="true" />
      <Nav />
      <main>
        <Hero photo="/samples/hero.jpg" />
        <LightTable />
        <ContactSheet />
        <Method />
        <Bench />
        <Api />
      </main>
      <Footer />
    </>
  );
}
