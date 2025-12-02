import re
import sympy as sp
import torch
from typing import Dict, Any, Optional
import pint

# Initialize unit registry
ureg = pint.UnitRegistry()

class CompositeVerifier:
    """Multi-objective verifier with numeric, units, algebraic, and self-consistency checks"""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "numeric": 1.0,
            "units": 0.5,
            "self_consistency": 0.3,
            "algebraic": 0.2
        }
        self.num_pattern = re.compile(r"(-?\d+(?:\.\d+)?)")

    def extract_answer(self, text: str) -> Optional[sp.Basic]:
        """Extract numeric answer from text"""
        matches = list(self.num_pattern.finditer(text))
        if not matches:
            return None
        try:
            return sp.nsimplify(matches[-1].group(1))
        except:
            return None

    def check_units(self, text: str, expected_unit: str = None) -> float:
        """Check dimensional consistency"""
        try:
            # Extract quantities with units
            quantities = re.findall(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", text)
            if not quantities:
                return 1.0

            # Parse with pint
            parsed = []
            for val, unit in quantities:
                try:
                    q = ureg.Quantity(float(val), unit)
                    parsed.append(q)
                except:
                    pass

            if len(parsed) < 2:
                return 1.0

            # Check dimensional consistency
            dims = [q.dimensionality for q in parsed]
            if len(set(str(d) for d in dims)) == 1:
                return 0.0  # All same dimension, good
            else:
                return 0.5  # Mixed dimensions, partial penalty
        except:
            return 1.0

    def check_algebraic_form(self, pred_text: str, gold_text: str) -> float:
        """Check if algebraic structure matches"""
        try:
            # Extract algebraic expressions
            pred_expr = self.extract_expression(pred_text)
            gold_expr = self.extract_expression(gold_text)

            if pred_expr is None or gold_expr is None:
                return 1.0

            # Check structural similarity
            if sp.simplify(pred_expr - gold_expr) == 0:
                return 0.0

            # Check if same variables
            pred_vars = pred_expr.free_symbols
            gold_vars = gold_expr.free_symbols
            if pred_vars == gold_vars:
                return 0.3

            return 1.0
        except:
            return 1.0

    def extract_expression(self, text: str) -> Optional[sp.Basic]:
        """Extract algebraic expression from text"""
        # Look for patterns like "x = ..." or "answer = ..."
        expr_match = re.search(r"(?:=|is)\s*([^.]+)", text)
        if expr_match:
            try:
                return sp.sympify(expr_match.group(1))
            except:
                pass
        return None

    def check_self_consistency(self, model, tokenizer, prompt: str,
                              n_samples: int = 3, temperature: float = 0.7) -> float:
        """Check if model gives consistent answers across samples"""
        answers = []
        for _ in range(n_samples):
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=128,
                                        temperature=temperature, do_sample=True)
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = self.extract_answer(text)
                if answer is not None:
                    answers.append(answer)

        if len(answers) < 2:
            return 1.0

        # Check variance in answers
        numeric_answers = [float(sp.N(a)) for a in answers]
        mean_val = sum(numeric_answers) / len(numeric_answers)
        variance = sum((x - mean_val)**2 for x in numeric_answers) / len(numeric_answers)

        # Normalize variance to [0, 1]
        return min(1.0, variance / (mean_val**2 + 1e-6))

    def compute_loss(self, pred_text: str, gold_text: str,
                    model=None, tokenizer=None, prompt: str = None) -> Dict[str, float]:
        """Compute multi-objective verification loss"""
        losses = {}

        # Numeric loss
        pred_num = self.extract_answer(pred_text)
        gold_num = self.extract_answer(gold_text)
        if pred_num is not None and gold_num is not None:
            try:
                losses["numeric"] = float((sp.N(pred_num) - sp.N(gold_num))**2)
            except:
                losses["numeric"] = 1.0
        else:
            losses["numeric"] = 1.0

        # Units loss
        losses["units"] = self.check_units(pred_text)

        # Algebraic form loss
        losses["algebraic"] = self.check_algebraic_form(pred_text, gold_text)

        # Self-consistency loss
        if model is not None and tokenizer is not None and prompt is not None:
            losses["self_consistency"] = self.check_self_consistency(
                model, tokenizer, prompt
            )
        else:
            losses["self_consistency"] = 0.0

        # Weighted total
        total = sum(self.weights.get(k, 0) * v for k, v in losses.items())
        losses["total"] = total

        return losses

