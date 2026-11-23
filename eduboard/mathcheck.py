"""Verification symbolique avec sympy.

Le LLM peut mal lire un exposant ou se tromper dans un calcul : avant
d'expliquer une formule, l'agent la fait verifier ici. C'est un outil
deterministe, donc on peut lui faire confiance plus qu'au modele.
"""
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
OPERATIONS = ("simplify", "equals", "solve", "derivative", "integral", "evaluate")


class MathError(ValueError):
    pass


def _parse(expr: str):
    expr = (expr or "").strip()
    if not expr:
        raise MathError("expression vide")
    # quelques ecritures de tableau courantes
    expr = expr.replace("·", "*").replace("×", "*").replace("÷", "/").replace("−", "-")
    try:
        return parse_expr(expr, transformations=TRANSFORMS, evaluate=True)
    except Exception as e:  # sympy leve tout et n'importe quoi
        raise MathError(f"expression non reconnue : {expr!r} ({type(e).__name__})") from e


def _split_equation(expr: str):
    if expr.count("=") == 1:
        lhs, rhs = expr.split("=")
        return _parse(lhs), _parse(rhs)
    return _parse(expr), sp.Integer(0)


def check_math(operation: str, expression: str, expected: str = "", variable: str = "x") -> str:
    if operation not in OPERATIONS:
        raise MathError(f"operation inconnue : {operation} (choix : {', '.join(OPERATIONS)})")

    var = sp.Symbol(variable or "x")
    # TODO: ajouter un timeout, certaines integrales font tourner sympy tres longtemps

    if operation == "simplify":
        return f"simplifie : {sp.simplify(_parse(expression))}"

    if operation == "equals":
        if "=" in expression and not expected:
            a, b = _split_equation(expression)
        else:
            a, b = _parse(expression), _parse(expected)
        diff = sp.simplify(a - b)
        if diff == 0:
            return "VRAI : les deux expressions sont egales"
        return f"FAUX : difference = {diff}"

    if operation == "solve":
        lhs, rhs = _split_equation(expression)
        sols = sp.solve(sp.Eq(lhs, rhs), var)
        return f"solutions pour {var} : {sols}" if sols else "aucune solution trouvee"

    if operation == "derivative":
        d = sp.simplify(sp.diff(_parse(expression), var))
        msg = f"d/d{var} = {d}"
        if expected:
            ok = sp.simplify(d - _parse(expected)) == 0
            msg += " | resultat attendu " + ("CORRECT" if ok else f"INCORRECT (attendu {expected})")
        return msg

    if operation == "integral":
        res = sp.integrate(_parse(expression), var)
        msg = f"primitive = {res} + C"
        if expected:
            # deux primitives correctes different d'une constante
            ok = sp.simplify(sp.diff(_parse(expected), var) - _parse(expression)) == 0
            msg += " | primitive proposee " + ("CORRECTE" if ok else "INCORRECTE")
        return msg

    # evaluate
    val = _parse(expression).evalf(8)
    return f"valeur = {val}"
