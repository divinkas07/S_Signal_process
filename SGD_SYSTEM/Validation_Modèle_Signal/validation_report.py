"""
Ir divinkas - SYNAPTIC Lab
SGD System - Validation Report Utility

Fournit des outils pour formater et afficher les resultats des tests de validation.
"""

class ValidationReport:
    """
    Gere l'affichage et la compilation des resultats de validation.
    """
    def __init__(self, title: str):
        self.title = title
        self.results = []
        self.width = 70

    def add_result(self, test_name: str, passed: bool, details: str):
        """Ajoute un resultat de test a la liste."""
        self.results.append({
            "name": test_name,
            "passed": passed,
            "details": details
        })
        self._print_immediate(test_name, passed, details)

    def _print_immediate(self, test_name: str, passed: bool, details: str):
        """Affiche immediatement le resultat d'un test pendant l'execution."""
        icon = "[PASS]" if passed else "[FAIL]"
        separator = "=" * self.width
        sub_sep = "-" * self.width
        
        print(f"\n{separator}")
        print(f"  {icon}  |  {test_name}")
        print(f"{sub_sep}")
        # Indenter les détails pour la lisibilité
        indented_details = "\n".join([f"  {line}" for line in details.split("\n")])
        print(indented_details)
        print(f"{separator}")

    def print_summary(self):
        """Affiche un bilan global de tous les tests effectués."""
        n_pass = sum(1 for r in self.results if r["passed"])
        n_total = len(self.results)
        all_pass = n_pass == n_total

        print("\n" + "#" * self.width)
        print(f"  BILAN : {n_pass}/{n_total} tests reussis")
        print("#" * self.width)

        for res in self.results:
            icon = "(V)" if res["passed"] else "(X)"
            print(f"  {icon}  {res['name']}")

        if all_pass:
            print(f"\n  GO - Tous les criteres sont satisfaits.")
            print("  ->  Le modele est valide.")
        else:
            print(f"\n  NO-GO - Certains criteres ne sont pas satisfaits.")
            print("  ! Action requise : Verifier les echecs ci-dessus.")
        print()

def get_report(title: str) -> ValidationReport:
    """Factory pour obtenir une instance de rapport."""
    return ValidationReport(title)
