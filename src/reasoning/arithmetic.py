"""Labelled reconstruction of the paper's masked-RPN arithmetic task.

Source-unobserved choices are deliberate: uniform Catalan shapes, exact integer
arithmetic, lexicographically first valid operator labels, a versioned group
hash split and digit-wise result encoding. No source dataset is reproduced.
"""
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random

OPERATORS = ("+", "-", "*", "/")
PAD, MASK, SEPARATOR = 0, 11, 12
VOCAB_SIZE, SEQUENCE_LENGTH, MAX_OPERATORS = 13, 19, 7
SPLIT_SALT = bytes.fromhex("67726164706174682d61726974686d657469632d70696c6f742d7631").decode()


def operate(left, right, operator):
    if operator == 0:
        return left + right
    if operator == 1:
        return left - right
    if operator == 2:
        return left * right
    if operator == 3:
        if right == 0 or left % right:
            raise ValueError("division must be nonzero and exact")
        return left // right
    raise ValueError("unknown operator")


def evaluate_rpn(structure, operands, operators):
    """Independent execution interface; rejects malformed or partial programs."""
    stack = []
    numbers, choices = iter(operands), iter(operators)
    try:
        for token in structure:
            if token == "n":
                value = next(numbers)
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError("operand must be an integer")
                stack.append(value)
            elif token == "o":
                if len(stack) < 2:
                    raise ValueError("RPN stack underflow")
                right, left = stack.pop(), stack.pop()
                stack.append(operate(left, right, next(choices)))
            else:
                raise ValueError("unknown structural token")
    except StopIteration as error:
        raise ValueError("missing operand or operator") from error
    if len(stack) != 1:
        raise ValueError("RPN must finish with one value")
    sentinel = object()
    if next(numbers, sentinel) is not sentinel or next(choices, sentinel) is not sentinel:
        raise ValueError("extra operands or operators")
    return stack[0]


@lru_cache(maxsize=8)
def shapes(operand_count):
    if not 1 <= operand_count <= 8:
        raise ValueError("shape size out of range")
    if operand_count == 1:
        return (("n",),)
    return tuple(
        left + right + ("o",)
        for left_count in range(1, operand_count)
        for left in shapes(left_count)
        for right in shapes(operand_count - left_count)
    )


def canonical_solutions(structure, operands):
    """Dynamic programming maps integer results to the first operator tuple.

    Multiple assignments yielding the same result are merged deterministically.
    This removes arbitrary sampled-label ambiguity, but differs from an unseen
    author tie policy. Tests compare this definition with exhaustive execution.
    """
    stack, cursor = [], 0
    for token in structure:
        if token == "n":
            if cursor >= len(operands):
                raise ValueError("missing operand")
            stack.append({operands[cursor]: ()})
            cursor += 1
        elif token == "o":
            if len(stack) < 2:
                raise ValueError("RPN stack underflow")
            right, left = stack.pop(), stack.pop()
            values = {}
            for left_value, left_ops in left.items():
                for right_value, right_ops in right.items():
                    for operator in range(4):
                        try:
                            value = operate(left_value, right_value, operator)
                        except ValueError:
                            continue
                        assignment = left_ops + right_ops + (operator,)
                        previous = values.get(value)
                        if previous is None or assignment < previous:
                            values[value] = assignment
            stack.append(values)
        else:
            raise ValueError("unknown structural token")
    if len(stack) != 1 or cursor != len(operands):
        raise ValueError("malformed RPN or operand count")
    return stack[0]


def multiset_key(operands):
    if not 3 <= len(operands) <= 8 or any(type(v) is not int or not 1 <= v <= 9 for v in operands):
        raise ValueError("operands must contain three to eight digits1..9")
    return ",".join(map(str, sorted(operands)))


def group_split(operands):
    key = multiset_key(operands)
    bucket = int(hashlib.sha256((SPLIT_SALT + ":" + key).encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "id_dev" if bucket < 90 else "ood_final"


def encode(structure, operands, result, operator_ids):
    if type(result) is not int or not 0 <= result <= 201:
        raise ValueError("result outside declared pilot ranges")
    if not 3 <= len(operands) <= 8 or len(operator_ids) != len(operands) - 1:
        raise ValueError("operand/operator count")
    if evaluate_rpn(structure, operands, operator_ids) != result:
        raise ValueError("label does not compute the observed result")
    tokens, positions = [], []
    cursor = 0
    for marker in structure:
        if marker == "n":
            digit = operands[cursor]
            if type(digit) is not int or not 1 <= digit <= 9:
                raise ValueError("operand out of range")
            tokens.append(digit + 1)
            cursor += 1
        else:
            positions.append(len(tokens))
            tokens.append(MASK)
    tokens += [SEPARATOR] + [int(character) + 1 for character in str(result)]
    if len(tokens) > SEQUENCE_LENGTH:
        raise ValueError("encoding exceeds19 tokens")
    count = len(operator_ids)
    return {
        "token_ids": tokens + [PAD] * (SEQUENCE_LENGTH - len(tokens)),
        "target_operator_ids": list(operator_ids) + [0] * (MAX_OPERATORS - count),
        "target_positions": positions + [0] * (MAX_OPERATORS - count),
        "target_mask": [True] * count + [False] * (MAX_OPERATORS - count),
    }


def build_dataset(output_directory, counts, seed=330034, max_proposals=2000000):
    """Create fixed independent groups before any model outcome is observed."""
    output_directory = Path(output_directory)
    if set(counts) != {"train", "id_dev", "ood_final"} or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError("three positive declared split counts required")
    output_directory.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    seen, accepted = set(), dict.fromkeys(counts, 0)
    groups = {key: set() for key in counts}
    lengths = {key: dict.fromkeys(range(3, 9), 0) for key in counts}
    handles = {split: (output_directory / (split + ".jsonl")).open("w") for split in counts}
    proposals = 0
    try:
        while accepted != counts:
            proposals += 1
            if proposals > max_proposals:
                raise RuntimeError("bounded generation did not fill all declared splits")
            length = rng.randint(3, 8)
            operands = tuple(rng.randint(1, 9) for _ in range(length))
            split = group_split(operands)
            if accepted[split] >= counts[split]:
                continue
            structure = rng.choice(shapes(length))
            solutions = canonical_solutions(structure, operands)
            lower, upper = (102, 201) if split == "ood_final" else (0, 101)
            values = sorted(value for value in solutions if lower <= value <= upper)
            if not values:
                continue
            # Uniform over attainable values within the accepted range; this is
            # explicitly a reconstruction choice, not the source sampler.
            result = rng.choice(values)
            encoded = encode(structure, operands, result, solutions[result])
            digest = hashlib.sha256(json.dumps(encoded["token_ids"], separators=(",", ":")).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            key = multiset_key(operands)
            row = dict(encoded, example_id=digest, multiset=key, structure=list(structure), operands=list(operands), result=result, split=split)
            handles[split].write(json.dumps(row, separators=(",", ":")) + "\n")
            accepted[split] += 1
            groups[split].add(key)
            lengths[split][length] += 1
        files = {}
        for handle in handles.values():
            handle.flush()
        for split in counts:
            path = output_directory / (split + ".jsonl")
            files[split] = {"path": str(path), "count": accepted[split], "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for left, right in itertools.combinations(counts, 2):
            if groups[left] & groups[right]:
                raise AssertionError("multiset leaked across splits")
        manifest = {
            "schema": "reasoning-arithmetic-reconstruction-v1", "seed": seed,
            "files": files, "proposals": proposals, "unique_input_count": len(seen),
            "multiset_counts": {key: len(value) for key, value in groups.items()},
            "operand_count_histograms": lengths, "group_overlap_count": 0,
            "split_rule": "salted SHA256 multisets:80%train,10%IDdev,10%OODfinal",
            "train_and_ID_value_range": [0, 101], "OOD_value_range": [102, 201],
            "tie_rule": "lexicographically first valid operator IDs(+,-,*,/) for each maskedRPN/result",
            "sampling": "uniform length3..8/operands1..9/Catalan shapes; uniform attainable in-range value; reject duplicates",
            "vocab_size": VOCAB_SIZE, "sequence_length": SEQUENCE_LENGTH,
            "source_fidelity": "labelled reconstruction; source sampler/tokenizer/tie policy and exact train sizes unobserved",
        }
        (output_directory / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    finally:
        for handle in handles.values():
            handle.close()


def load_records(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]
