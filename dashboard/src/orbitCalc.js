export function firstPositiveInteger(...values) {
  for (const value of values) {
    const parsed = Number(value);
    if (Number.isFinite(parsed) && parsed >= 1) {
      return Math.floor(parsed);
    }
  }
  return 1;
}

export function buildOrbit(seed, maxSteps = 18) {
  let current = typeof seed === "bigint" ? seed : BigInt(seed || 1);
  const frames = [];

  for (let step = 0; step < maxSteps; step += 1) {
    frames.push({
      step,
      value: current.toString()
    });
    if (current === 1n) {
      break;
    }
    current = current % 2n === 0n ? current / 2n : (3n * current) + 1n;
  }

  return frames;
}

export function twoAdicValuation(value) {
  let current = BigInt(value || 0);
  let power = 0;
  while (current > 0n && current % 2n === 0n) {
    current /= 2n;
    power += 1;
  }
  return power;
}

export function nextCollatzValue(value) {
  const current = BigInt(value || 1);
  if (current % 2n === 0n) {
    const next = current / 2n;
    return {
      parity: "even",
      expression: `a_(k+1) = a_k / 2 = ${next}`,
      latex: `a_{k+1} = a_k / 2 = ${next}`,
      next: next.toString(),
      rule: `${current} ≡ 0 (mod 2)`,
      acceleration: ""
    };
  }
  const next = (3n * current) + 1n;
  const valuation = twoAdicValuation(next);
  const compressed = next / (2n ** BigInt(valuation));
  return {
    parity: "odd",
    expression: `a_(k+1) = 3a_k + 1 = ${next}`,
    latex: `a_{k+1} = 3 \\cdot a_k + 1 = ${next}`,
    next: next.toString(),
    rule: `${current} ≡ 1 (mod 2)`,
    acceleration: `${next} = 2^{${valuation}} \\cdot ${compressed}`
  };
}

export function orbitStats(orbit) {
  if (orbit.length === 0) {
    return { evenSteps: 0, oddSteps: 0, maxValue: "1" };
  }
  let evenSteps = 0;
  let oddSteps = 0;
  let maxValue = 0n;
  for (const item of orbit) {
    const value = BigInt(item.value);
    if (value % 2n === 0n) {
      evenSteps += 1;
    } else {
      oddSteps += 1;
    }
    if (value > maxValue) {
      maxValue = value;
    }
  }
  return {
    evenSteps,
    oddSteps,
    maxValue: maxValue.toString()
  };
}

export function buildMathTrace(orbit, startIndex, kernel, limit = 5) {
  if (orbit.length === 0) {
    return [];
  }
  const rows = [];
  const firstIndex = Math.max(0, Math.min(startIndex, Math.max(0, orbit.length - 1)));

  for (let offset = 0; offset < limit; offset += 1) {
    const index = firstIndex + offset;
    if (index >= orbit.length) {
      break;
    }

    const current = BigInt(orbit[index].value);
    if (current === 1n) {
      rows.push({
        key: `${index}-terminal`,
        step: index,
        formula: `a_${index} = 1`,
        latex: `a_{${index}} = 1`,
        note: "The trace has entered the trivial loop 1 → 4 → 2 → 1.",
        acceleration: ""
      });
      break;
    }

    if (current % 2n === 0n) {
      const next = current / 2n;
      rows.push({
        key: `${index}-even`,
        step: index,
        formula: `a_${index + 1} = a_${index} / 2 = ${next}`,
        latex: `a_{${index + 1}} = a_{${index}} / 2 = ${next}`,
        note: `${current} ≡ 0 (mod 2) — even branch, halve.`,
        acceleration: ""
      });
      continue;
    }

    const lifted = (3n * current) + 1n;
    const valuation = twoAdicValuation(lifted);
    const compressed = lifted / (2n ** BigInt(valuation));
    rows.push({
      key: `${index}-odd`,
      step: index,
      formula: `a_${index + 1} = 3a_${index} + 1 = ${lifted}`,
      latex: `a_{${index + 1}} = 3 \\cdot a_{${index}} + 1 = ${lifted}`,
      note: `${current} ≡ 1 (mod 2) — odd branch, expand.`,
      acceleration:
        kernel === "cpu-accelerated"
          ? `${lifted} = 2^{${valuation}} \\cdot ${compressed} → compressed to ${compressed}.`
          : `${lifted} = 2^{${valuation}} \\cdot ${compressed}.`
    });
  }

  return rows;
}

export function summarizeMetric(metric) {
  if (metric === "max_total_stopping_time") {
    return {
      symbol: "sigma",
      label: "total stopping time",
      definition: "steps required to reach 1"
    };
  }
  if (metric === "max_stopping_time") {
    return {
      symbol: "tau",
      label: "descent time",
      definition: "first time the orbit falls below its seed"
    };
  }
  return {
    symbol: "E",
    label: "peak excursion",
    definition: "largest value reached on the orbit"
  };
}

const METRIC_LATEX = {
  sigma: "\\sigma",
  tau: "\\tau",
  E: "E"
};

export function buildMetricSummary(run) {
  if (!run?.metrics) {
    return [];
  }
  return ["max_total_stopping_time", "max_stopping_time", "max_excursion"]
    .map((metric) => {
      const record = run.metrics?.[metric];
      if (!record || Number(record.n) < 1) {
        return null;
      }
      const details = summarizeMetric(metric);
      const sym = METRIC_LATEX[details.symbol] || details.symbol;
      return {
        key: metric,
        label: details.label,
        formula: `${details.symbol}(${record.n}) = ${record.value}`,
        latex: `${sym}(${record.n}) = ${record.value}`,
        definition: details.definition
      };
    })
    .filter(Boolean);
}

export function buildRecordTape(run) {
  const records = Array.isArray(run?.metrics?.sample_records) ? run.metrics.sample_records : [];
  return [...records]
    .reverse()
    .slice(0, 6)
    .map((record, index) => {
      const details = summarizeMetric(record.metric);
      const sym = METRIC_LATEX[details.symbol] || details.symbol;
      return {
        key: `${record.metric}-${record.n}-${record.value}-${index}`,
        label: details.label,
        formula: `${details.symbol}(${record.n}) = ${record.value}`,
        latex: `${sym}(${record.n}) = ${record.value}`,
        definition: details.definition
      };
    });
}

export function runProgress(run) {
  if (!run) {
    return { processed: 0, total: 0, percent: 0 };
  }
  const total = Math.max(0, Number(run.range_end) - Number(run.range_start) + 1);
  const processed = Math.max(0, Number(run.metrics?.processed || run.checkpoint?.last_processed || 0));
  const percent = total > 0 ? Math.min(100, (processed / total) * 100) : 0;
  return { processed, total, percent };
}

export function runLiveDetails(run) {
  const p = runProgress(run);
  if (!run) return { ...p, speed: 0, eta: "", currentSeed: 0, maxTST: null, maxExc: null };

  // Speed + ETA from elapsed time
  let speed = 0;
  let eta = "";
  if (run.started_at && p.processed > 0) {
    const elapsed = (Date.now() - new Date(run.started_at).getTime()) / 1000;
    if (elapsed > 0) {
      speed = p.processed / elapsed;
      const remaining = p.total - p.processed;
      if (speed > 0 && remaining > 0) {
        const secs = remaining / speed;
        if (secs < 60) eta = `${Math.round(secs)}s`;
        else if (secs < 3600) eta = `${Math.round(secs / 60)}m`;
        else if (secs < 86400) eta = `${(secs / 3600).toFixed(1)}h`;
        else eta = `${(secs / 86400).toFixed(1)}d`;
      } else if (remaining <= 0) {
        eta = "done";
      }
    }
  }

  const currentSeed = Number(run.checkpoint?.next_value || run.checkpoint?.last_processed || 0);
  const maxTST = run.metrics?.max_total_stopping_time || null;
  const maxExc = run.metrics?.max_excursion || null;

  return { ...p, speed, eta, currentSeed, maxTST, maxExc };
}

export function describeCapability(capability) {
  if (!capability) {
    return "No capability record exposed yet.";
  }
  const executionReady = capability.metadata?.execution_ready;
  if (capability.kind === "gpu") {
    return executionReady
      ? `GPU path is executable with kernels: ${(capability.supported_kernels || []).join(", ")}.`
      : "GPU is detected but has no executable kernel yet, so the dashboard cannot dispatch real runs to it.";
  }
  return executionReady
    ? `CPU worker can execute: ${(capability.supported_kernels || []).join(", ")}.`
    : "CPU capability is visible but not executable yet.";
}

export function orbitSeedFromRun(run) {
  return firstPositiveInteger(
    run?.checkpoint?.last_processed,
    run?.checkpoint?.next_value,
    run?.metrics?.max_total_stopping_time?.n,
    run?.metrics?.max_excursion?.n,
    run?.range_start
  );
}

export function orbitSeedLabel(run) {
  if (!run) {
    return "no run selected";
  }
  if (run.status === "running" && Number(run.checkpoint?.last_processed) >= 1) {
    return `live checkpoint ${run.checkpoint.last_processed}`;
  }
  if (Number(run.metrics?.max_total_stopping_time?.n) >= 1) {
    return `record seed ${run.metrics.max_total_stopping_time.n}`;
  }
  return `seed ${run.range_start}`;
}

export function describeOrbitSeed(run, orbitSeed) {
  const seed = BigInt(orbitSeed || 1);
  const valuation = twoAdicValuation(seed);
  const sourceLabel = run?.status === "running" && Number(run?.checkpoint?.last_processed) >= 1
    ? `checkpoint input n = ${run.checkpoint.last_processed}`
    : Number(run?.checkpoint?.last_processed) >= 1
      ? `saved checkpoint seed n = ${run.checkpoint.last_processed}`
      : Number(run?.metrics?.max_total_stopping_time?.n) >= 1
        ? `record seed n = ${run.metrics.max_total_stopping_time.n}`
        : `range seed n = ${run?.range_start ?? orbitSeed}`;
  const reconstructionNote = run?.status === "running"
    ? "The UI recomputes the next visible Collatz iterates from the latest real checkpoint input. This is exact for that seed, but it is not yet a direct inner-kernel trace dump."
    : "The UI is replaying the next visible Collatz iterates from a saved real seed. The formulas are exact for that seed, but they are reconstructed from stored run data.";
  const parityNote = valuation > 0
    ? `This seed is divisible by 2^${valuation}, so the first ${valuation} visible step${valuation === 1 ? "" : "s"} are immediate halvings. Similar leading blocks are expected when checkpoints land on nearby even inputs.`
    : "This seed is odd, so the next visible step should be a 3n + 1 expansion before any halving compression appears.";
  return {
    sourceLabel,
    detail: `${reconstructionNote} ${parityNote}`
  };
}
