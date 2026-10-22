"""Dependency-free, delayed-feedback regression reference implementation.

No policy receives task identities, regime boundaries, unavailable labels, or RNG.
The runner, not the policy, owns pending targets and scoring.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, asdict


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def finite(values):
    if any(not math.isfinite(v) for v in values):
        raise ValueError("nonfinite input or state")


@dataclass(frozen=True)
class Config:
    kind: str = "bank"
    eta: float = 0.5
    alpha: float = 0.05
    capacity: int = 4
    interval: int = 32
    minimum: int = 8
    gate: float = 0.9
    suppress: bool = True
    uniform: bool = False
    forgetting: float = 0.99
    cache: int = 32
    neighbors: int = 3

    def __post_init__(self):
        if self.kind not in {"bank", "nlms", "adam", "rls", "knn", "replay", "frozen", "last"}:
            raise ValueError("unknown method")
        if not 0 < self.eta < 2 or not 0 < self.alpha <= 1 or not 0 < self.gate <= 1:
            raise ValueError("invalid learning/gating parameter")
        if not 0 < self.forgetting <= 1:
            raise ValueError("invalid forgetting")
        for n in (self.capacity, self.interval, self.minimum, self.cache, self.neighbors):
            if type(n) is not int or n <= 0:
                raise ValueError("counts must be positive integers")


class Policy:
    def __init__(self, dimension, config):
        if type(dimension) is not int or dimension < 1:
            raise ValueError("invalid dimension")
        self.d = dimension + 1
        self.config = config
        self.w = [0.0] * self.d
        self.bank = []
        self.next_id = 0
        self.loss = 0.0
        self.count = 0
        self.cache = []
        self.last = 0.0
        self.first = [0.0] * self.d if config.kind == "adam" else []
        self.second = [0.0] * self.d if config.kind == "adam" else []
        self.cov = [[10.0 * (i == j) for j in range(self.d)] for i in range(self.d)] if config.kind == "rls" else []
        self.usage = {k: 0 for k in ("predictions", "labels", "updates", "gradients", "dot_products", "coordinate_ops", "archive_writes", "cache_writes", "evictions", "memory_predictions", "skips", "loss_scores")}

    def x(self, features):
        if len(features) != self.d - 1:
            raise ValueError("dimension mismatch")
        finite(features)
        return list(features) + [1.0]

    def dot(self, a, b):
        self.usage["dot_products"] += 1
        self.usage["coordinate_ops"] += 2 * len(a)
        return sum(v * w for v, w in zip(a, b))

    def winner(self):
        eligible = [e for e in self.bank if e["count"] >= self.config.minimum and e["loss"] < self.config.gate * self.loss]
        return min(eligible, key=lambda e: (e["loss"], e["id"])) if eligible else None

    def predict(self, features):
        x = self.x(features)
        c = self.config
        active = self.dot(self.w, x)
        predictions = {str(e["id"]): self.dot(e["w"], x) for e in self.bank}
        winner = self.winner()
        selected = winner["id"] if winner else -1
        prediction = predictions[str(selected)] if selected >= 0 else active
        if c.uniform and predictions:
            prediction = (active + sum(predictions.values())) / (len(predictions) + 1)
            selected = -2
        if c.kind == "last":
            prediction = self.last
        if c.kind == "knn" and self.cache:
            distances = [(sum((a - b) ** 2 for a, b in zip(x, z)), i, y) for i, (z, y) in enumerate(self.cache)]
            self.usage["coordinate_ops"] += 3 * self.d * len(self.cache)
            closest = sorted(distances)[:c.neighbors]
            prediction = sum(z[2] for z in closest) / len(closest)
        finite([prediction, active] + list(predictions.values()))
        self.usage["predictions"] += 1
        self.usage["memory_predictions"] += int(selected != -1)
        return prediction, {"active": active, "experts": predictions, "selected": selected}

    def nlms(self, x, y):
        error = y - self.dot(self.w, x)
        scale = self.config.eta * error / (1e-8 + sum(v * v for v in x))
        self.w = [w + scale * v for w, v in zip(self.w, x)]
        self.usage["updates"] += 1
        self.usage["gradients"] += 1
        self.usage["coordinate_ops"] += 5 * self.d + 3

    def observe(self, features, target, ticket):
        x = self.x(features)
        finite([target, ticket["active"]] + list(ticket["experts"].values()))
        c = self.config
        self.usage["labels"] += 1
        loss = (ticket["active"] - target) ** 2
        self.loss = loss if self.count == 0 else (1-c.alpha)*self.loss + c.alpha*loss
        self.count += 1
        self.usage["loss_scores"] += 1
        self.usage["coordinate_ops"] += 6
        for e in self.bank:
            if str(e["id"]) in ticket["experts"]:
                loss = (ticket["experts"][str(e["id"])] - target) ** 2
                e["loss"] = loss if e["count"] == 0 else (1-c.alpha)*e["loss"] + c.alpha*loss
                e["count"] += 1
                self.usage["loss_scores"] += 1
                self.usage["coordinate_ops"] += 6
        if c.kind in {"frozen", "last"}:
            self.last = target
        elif c.kind == "knn":
            self.remember(x, target)
        elif c.kind == "adam":
            error = self.dot(self.w, x) - target
            grad = [error * v for v in x]
            scale = min(1.0, 10 / max(1e-30, math.sqrt(sum(v*v for v in grad))))
            t = self.usage["updates"] + 1
            for i in range(self.d):
                g = grad[i] * scale
                self.first[i] = 0.9*self.first[i] + 0.1*g
                self.second[i] = 0.999*self.second[i] + 0.001*g*g
                self.w[i] -= c.eta*(self.first[i]/(1-0.9**t))/(math.sqrt(self.second[i]/(1-0.999**t))+1e-8)
            self.usage["updates"] += 1
            self.usage["gradients"] += 1
            self.usage["coordinate_ops"] += 22*self.d
        elif c.kind == "rls":
            px = [self.dot(row, x) for row in self.cov]
            denom = c.forgetting + self.dot(x, px)
            if denom <= 0:
                raise ArithmeticError("RLS lost positive covariance")
            error = target - self.dot(self.w, x)
            self.w = [w + p*error/denom for w,p in zip(self.w, px)]
            self.cov = [[(self.cov[i][j]-px[i]*px[j]/denom)/c.forgetting for j in range(self.d)] for i in range(self.d)]
            self.usage["updates"] += 1
            self.usage["coordinate_ops"] += 4*self.d*self.d+3*self.d
        elif c.kind == "bank" and c.suppress and self.winner() is not None:
            self.usage["skips"] += 1
        else:
            self.nlms(x, target)
            if c.kind == "replay":
                if self.cache:
                    self.nlms(*self.cache[0])
                self.remember(x, target)
            if c.kind == "bank" and self.usage["updates"] % c.interval == 0:
                self.bank.append({"id": self.next_id, "w": self.w[:], "count": 0, "loss": 0.0})
                self.next_id += 1
                self.usage["archive_writes"] += 1
                if len(self.bank) > c.capacity:
                    self.bank.pop(0)
                    self.usage["evictions"] += 1
        finite(self.w + self.first + self.second + [z for row in self.cov for z in row] + [self.loss])

    def remember(self, x, y):
        self.cache.append([x, y])
        self.usage["cache_writes"] += 1
        if len(self.cache) > self.config.cache:
            self.cache.pop(0)
            self.usage["evictions"] += 1

    def numeric_bytes(self):
        return 8*(self.d + len(self.first)+len(self.second)+len(self.cov)*self.d + len(self.bank)*(self.d+3) + len(self.cache)*(self.d+1) + len(self.usage)+4)

    def snapshot(self):
        s = copy.deepcopy(self.__dict__)
        s["config"] = asdict(self.config)
        return s

    @classmethod
    def restore(cls, state, config, dimension):
        p = cls(dimension, config)
        if set(state) != set(p.snapshot()) or state["config"] != asdict(config) or state["d"] != dimension+1:
            raise ValueError("checkpoint/config mismatch")
        # Strict finite JSON validation also rejects NaN nested in caches and tickets.
        canonical(state)
        if len(state["w"]) != p.d or len(state["bank"]) > config.capacity or len(state["cache"]) > config.cache:
            raise ValueError("checkpoint shape/capacity mismatch")
        ids = [e["id"] for e in state["bank"]]
        if len(set(ids)) != len(ids) or any(len(e["w"]) != p.d or e["id"] >= state["next_id"] for e in state["bank"]):
            raise ValueError("checkpoint expert identity mismatch")
        for key,value in copy.deepcopy(state).items():
            if key != "config":
                setattr(p,key,value)
        return p


def synthetic(seed, condition="base", steps=960):
    """Eight regimes; no policy is told the boundary or target map."""
    if condition == "horizon":
        steps = 1920
    dimension = 16 if condition == "dimension" else 8
    noise = 0.2 if condition == "noise" else 0.05
    delay = 16 if condition == "delay" else 4
    r = random.Random(seed)
    maps = []
    for _ in range(8):
        w = [r.gauss(0,1) for _ in range(4)]
        norm = math.sqrt(sum(v*v for v in w))
        maps.append([v/norm for v in w])
    maps[1] = [-v for v in maps[0]]
    order = list(range(8)) if condition == "novel" else [0,1,0,2,1,0,2,0]
    rows = []
    for t in range(steps):
        regime = min(7,t//(steps//8))
        distractor = (6.0 if condition == "distractor" else 3.0) if regime % 2 else 1.0
        x = [r.gauss(0,1) for _ in range(4)] + [r.gauss(0,distractor) for _ in range(dimension-4)]
        y = sum(a*b for a,b in zip(x,maps[order[regime]])) + r.gauss(0,noise)
        rows.append({"step":t,"x":x,"y":y,"available":t+delay,"regime":regime,"return":regime>0 and order[regime] in order[:regime]})
    return rows


class Runner:
    def __init__(self, rows, config):
        if not rows:
            raise ValueError("empty task")
        for t,row in enumerate(rows):
            if row["step"] != t or row["available"] < t:
                raise ValueError("invalid task clock")
            finite(row["x"])
            if row["y"] is not None:
                finite([row["y"]])
        self.rows = rows
        self.task_digest = digest(rows)
        self.config = config
        self.policy = Policy(len(rows[0]["x"]), config)
        self.pending = []
        self.step = 0
        self.trace = []
        self.peak_bytes = 0

    def deliver(self,t):
        ready = [p for p in self.pending if p["available"] <= t]
        self.pending = [p for p in self.pending if p["available"] > t]
        for p in ready:
            self.policy.observe(p["x"], p["y"], p["ticket"])
        return [p["step"] for p in ready]

    def advance(self,stop=None):
        stop = len(self.rows) if stop is None else min(stop,len(self.rows))
        while self.step < stop:
            t = self.step
            r = self.rows[t]
            before = self.deliver(t)
            updates_before = self.policy.usage["updates"]
            pred,ticket = self.policy.predict(r["x"])
            if r["y"] is not None:
                self.pending.append({"step":t,"available":r["available"],"x":r["x"],"y":r["y"],"ticket":ticket})
            after = self.deliver(t)
            pending_bytes = sum(8*(len(p["x"])+6+2*len(p["ticket"]["experts"])) for p in self.pending)
            size = self.policy.numeric_bytes()+pending_bytes
            self.peak_bytes = max(self.peak_bytes,size)
            if size > 131072:
                raise MemoryError("algorithmic state cap exceeded")
            self.trace.append({"step":t,"prediction":pred,"target":r["y"],"squared_error":None if r["y"] is None else (pred-r["y"])**2,"regime":r["regime"],"return":r["return"],"labels_before":before,"labels_after":after,"updates_before":updates_before,"selected":ticket["selected"],"bank_ids":[e["id"] for e in self.policy.bank],"usage":dict(self.policy.usage),"state_bytes":size})
            self.step += 1
        return self.trace

    def snapshot(self):
        return {"schema":"neurogenesis.memory.v2","task_digest":self.task_digest,"config":asdict(self.config),"policy":self.policy.snapshot(),"pending":copy.deepcopy(self.pending),"step":self.step,"trace":copy.deepcopy(self.trace),"peak_bytes":self.peak_bytes}

    @classmethod
    def restore(cls, rows, config, state):
        runner = cls(rows,config)
        if set(state) != set(runner.snapshot()) or state["schema"] != "neurogenesis.memory.v2" or state["task_digest"] != runner.task_digest or state["config"] != asdict(config):
            raise ValueError("runner checkpoint identity mismatch")
        canonical(state)
        if state["step"] != len(state["trace"]) or not 0 <= state["step"] <= len(rows):
            raise ValueError("checkpoint clock mismatch")
        for p in state["pending"]:
            if not 0 <= p["step"] < state["step"] or p["available"] < state["step"] or p["y"] != rows[p["step"]]["y"] or p["x"] != rows[p["step"]]["x"]:
                raise ValueError("pending label mismatch")
        runner.policy = Policy.restore(state["policy"], config, len(rows[0]["x"]))
        for key in ("pending","step","trace","peak_bytes"):
            setattr(runner,key,copy.deepcopy(state[key]))
        return runner


def metrics(trace):
    scored = [r for r in trace if r["squared_error"] is not None]
    if not scored:
        raise ValueError("no scored predictions")
    errors = [r["squared_error"] for r in scored]
    finite(errors)
    switches,returns = [],[]
    age = 0
    for i,r in enumerate(trace):
        age = 0 if i == 0 or r["regime"] != trace[i-1]["regime"] else age+1
        if r["regime"] > 0 and age < 24 and r["squared_error"] is not None:
            switches.append(r["squared_error"])
            if r["return"]:
                returns.append(r["squared_error"])
    mean = lambda x: sum(x)/len(x) if x else None
    return {"mse":mean(errors),"scored":len(errors),"predictions":len(trace),"switch_mse":mean(switches),"return_mse":mean(returns),"usage":trace[-1]["usage"],"peak_state_bytes":max(r["state_bytes"] for r in trace)}
