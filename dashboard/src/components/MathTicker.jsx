import { useEffect, useRef, useState, memo } from "react";
import { MathNum } from "./ui.jsx";

function buildTickerEquations(orbit) {
  const equations = [];
  if (orbit.length > 0) {
    for (let i = 0; i < orbit.length; i++) {
      const val = BigInt(orbit[i].value);
      if (val === 1n) {
        equations.push({ key: `t-${i}`, step: i, value: "1", nextValue: "1", isEven: true, terminal: true });
        break;
      }
      const isEven = val % 2n === 0n;
      const nextVal = isEven ? val / 2n : 3n * val + 1n;
      equations.push({ key: `t-${i}`, step: i, value: String(val), nextValue: String(nextVal), isEven, terminal: false });
    }
  }
  if (equations.length === 0) {
    return [
      { key: "idle-a", idle: true },
      { key: "idle-b", idle: true, alt: true },
      { key: "idle-c", idle: true },
      { key: "idle-d", idle: true, alt: true }
    ];
  }
  return equations;
}

export default memo(function MathTicker({ run, orbit, frameIndex, isActive }) {
  const wrapRef = useRef(null);
  const trackRef = useRef(null);
  const segmentRef = useRef(null);
  const segmentWidthRef = useRef(0);
  const offsetRef = useRef(0);
  const animationRef = useRef(0);
  const lastTickRef = useRef(0);
  const pendingEquationsRef = useRef(null);
  const displayedSignatureRef = useRef("");
  const [segmentCopies, setSegmentCopies] = useState(4);
  const [displayedEquations, setDisplayedEquations] = useState(() => buildTickerEquations(orbit));
  const orbitSignature = orbit.length > 0 ? orbit.map((item) => item.value).join("|") : "idle";
  const pixelsPerSecond = displayedEquations.length < 8 ? 14 : displayedEquations.length < 16 ? 18 : 24;

  useEffect(() => {
    const nextEquations = buildTickerEquations(orbit);
    const nextIsIdle = nextEquations.every((equation) => equation.idle);
    if (displayedSignatureRef.current === "") {
      displayedSignatureRef.current = orbitSignature;
      setDisplayedEquations(nextEquations);
      return;
    }
    if (displayedSignatureRef.current === orbitSignature) {
      return;
    }
    if (nextIsIdle) {
      return;
    }
    pendingEquationsRef.current = {
      equations: nextEquations,
      signature: orbitSignature
    };
  }, [orbitSignature, orbit]);

  useEffect(() => {
    const segment = segmentRef.current;
    if (!segment) {
      return undefined;
    }

    const measure = () => {
      const previousWidth = segmentWidthRef.current;
      const width = segment.scrollWidth || segment.getBoundingClientRect().width || 0;
      const wrapWidth = wrapRef.current?.clientWidth || 0;
      if (previousWidth > 0 && width > 0) {
        const ratio = offsetRef.current / previousWidth;
        offsetRef.current = ratio * width;
      }
      segmentWidthRef.current = width;
      if (width > 0) {
        const neededCopies = Math.max(5, Math.ceil((wrapWidth || width) / width) + 4);
        setSegmentCopies((current) => (current === neededCopies ? current : neededCopies));
      }
      if (trackRef.current && width > 0) {
        if (offsetRef.current >= width) {
          offsetRef.current %= width;
        }
        trackRef.current.style.transform = `translate3d(${-offsetRef.current}px, 0, 0)`;
      }
    };

    measure();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(measure);
    observer?.observe(segment);
    window.addEventListener("resize", measure);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [run?.id, displayedEquations]);

  useEffect(() => {
    const track = trackRef.current;
    if (!track) {
      return undefined;
    }
    if (!isActive) {
      lastTickRef.current = 0;
      track.style.transform = `translate3d(${-offsetRef.current}px, 0, 0)`;
      return undefined;
    }

    let cancelled = false;
    lastTickRef.current = 0;

    const tick = (timestamp) => {
      if (cancelled) {
        return;
      }
      if (lastTickRef.current === 0) {
        lastTickRef.current = timestamp;
      }
      const deltaSeconds = (timestamp - lastTickRef.current) / 1000;
      lastTickRef.current = timestamp;
      const segmentWidth = segmentWidthRef.current;
      if (segmentWidth > 0) {
        offsetRef.current = (offsetRef.current + (deltaSeconds * pixelsPerSecond)) % segmentWidth;
        const pending = pendingEquationsRef.current;
        const seamDistance = Math.min(offsetRef.current, Math.abs(segmentWidth - offsetRef.current));
        if (pending && seamDistance < 6) {
          pendingEquationsRef.current = null;
          displayedSignatureRef.current = pending.signature;
          offsetRef.current = 0;
          setDisplayedEquations(pending.equations);
        }
        track.style.transform = `translate3d(${-offsetRef.current}px, 0, 0)`;
      }
      animationRef.current = window.requestAnimationFrame(tick);
    };

    animationRef.current = window.requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (animationRef.current) {
        window.cancelAnimationFrame(animationRef.current);
      }
    };
  }, [run?.id, pixelsPerSecond, isActive]);

  return (
    <div className="math-ticker-wrap" aria-hidden="true" ref={wrapRef}>
      <svg className="ticker-staff" viewBox="0 0 100 72" preserveAspectRatio="none">
        <line x1="0" y1="1" x2="100" y2="1" />
        <line x1="0" y1="18" x2="100" y2="18" />
        <line x1="0" y1="36" x2="100" y2="36" />
        <line x1="0" y1="54" x2="100" y2="54" />
        <line x1="0" y1="71" x2="100" y2="71" />
      </svg>
      <div className="math-ticker-track" ref={trackRef}>
        {Array.from({ length: segmentCopies }).map((_, copyIndex) => (
          <div
            key={`segment-${copyIndex}`}
            className="math-ticker-segment"
            ref={copyIndex === 0 ? segmentRef : undefined}
          >
            {displayedEquations.map((eq) =>
              eq.idle ? (
                <span key={`${copyIndex}-${eq.key}`} className="ticker-eq">
                  <span className="ticker-formula">
                    {eq.alt ? <>T(n) = 3n + 1, &nbsp; n {"\u2261"} 1 (mod 2)</> : <>T(n) = n / 2, &nbsp; n {"\u2261"} 0 (mod 2)</>}
                  </span>
                </span>
              ) : eq.terminal ? (
                <span key={`${copyIndex}-${eq.key}`} className="ticker-eq">
                  <span className="ticker-formula">a<sub>{eq.step}</sub> {"\u2192"} 1 {"\u220e"}</span>
                  <span className="ticker-values">orbit converged</span>
                </span>
              ) : (
                <span key={`${copyIndex}-${eq.key}`} className="ticker-eq">
                  <span className="ticker-formula">
                    T(a<sub>{eq.step}</sub>) = {eq.isEven ? <>a<sub>{eq.step}</sub> / 2</> : <>3{"\u00b7"}a<sub>{eq.step}</sub> + 1</>}
                  </span>
                  <span className="ticker-values">
                    <MathNum value={eq.value} /> <span className="ticker-arrow">{"\u2192"}</span> <MathNum value={eq.nextValue} />
                  </span>
                </span>
              )
            )}
          </div>
        ))}
      </div>
    </div>
  );
})
