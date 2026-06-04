from __future__ import annotations

import ast
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "qgan_latent"

EXPECTED_SCRIPTS = {
    "qgan-latent-pretrain": "qgan_latent.cli.latent_pretrain:main",
    "qgan-latent-train": "qgan_latent.cli.latent_train:main",
    "qgan-vqvae-pretrain": "qgan_latent.cli.vqvae_pretrain:main",
    "qgan-vqvae-train-prior": "qgan_latent.cli.vqvae_train_prior:main",
    "qgan-stage5-test-run": "qgan_latent.cli.stage5_test_run:main",
}
REMOVED_SCRIPTS = {"qgan-pretrain", "qgan-train", "qgan-train-prior"}
REMOVED_CLI_MODULES = {"pretrain.py", "train.py", "train_prior.py"}
REMOVED_ROOT_SCRIPTS = {"pretrain.py", "train.py", "train_prior.py"}
REMOVED_LEGACY_PACKAGES = {"datamodules", "models", "training", "utils"}


def _project_scripts() -> dict[str, str]:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["scripts"]


def _hydra_config_name(module_path: Path) -> str:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "main":
            continue
        for decorator in node.decorator_list:
            call = decorator
            if isinstance(call, ast.Call):
                for keyword in call.keywords:
                    if keyword.arg == "config_name" and isinstance(keyword.value, ast.Constant):
                        return str(keyword.value.value)
    raise AssertionError(f"No Hydra config_name found in {module_path}")


def test_stage4_console_scripts_are_explicit_and_legacy_scripts_removed() -> None:
    scripts = _project_scripts()
    assert scripts == EXPECTED_SCRIPTS
    assert REMOVED_SCRIPTS.isdisjoint(scripts)


def test_stage4_cli_modules_are_explicit_and_legacy_modules_removed() -> None:
    cli_dir = SRC / "cli"
    for module_name in [
        "latent_pretrain.py",
        "latent_train.py",
        "vqvae_pretrain.py",
        "vqvae_train_prior.py",
    ]:
        assert (cli_dir / module_name).is_file()
    for module_name in REMOVED_CLI_MODULES:
        assert not (cli_dir / module_name).exists()


def test_stage4_cli_modules_select_project_specific_configs() -> None:
    cli_dir = SRC / "cli"
    assert _hydra_config_name(cli_dir / "latent_pretrain.py") == "projects/qgan_expectation_values/pretrain"
    assert _hydra_config_name(cli_dir / "latent_train.py") == "projects/qgan_expectation_values/train"
    assert _hydra_config_name(cli_dir / "vqvae_pretrain.py") == "projects/tensor_prior_vqvae/pretrain"
    assert _hydra_config_name(cli_dir / "vqvae_train_prior.py") == "projects/tensor_prior_vqvae/train_prior"


def test_stage4_root_compatibility_scripts_removed() -> None:
    for script_name in REMOVED_ROOT_SCRIPTS:
        assert not (ROOT / script_name).exists()


def test_stage4_legacy_wrapper_packages_removed() -> None:
    for package_name in REMOVED_LEGACY_PACKAGES:
        assert not (SRC / package_name).exists()


def test_stage4_canonical_namespaces_remain_present() -> None:
    assert (SRC / "shared" / "representations" / "autoencoder.py").is_file()
    assert (SRC / "shared" / "training" / "pretrain_loop.py").is_file()
    assert (SRC / "workflows" / "qgan_expectation_values" / "training" / "gan_loop.py").is_file()
    assert (SRC / "workflows" / "tensor_prior_vqvae" / "training" / "mps_prior_loop.py").is_file()
