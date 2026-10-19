from src import config, modal_server


def _value_after(command, flag):
    return command[command.index(flag) + 1]


def test_vllm_command_takes_rank_names_and_length_from_config():
    command = modal_server.vllm_command(fast_boot=True)
    assert _value_after(command, "--max-lora-rank") == str(config.LORA_R)
    assert _value_after(command, "--served-model-name") == config.SERVED_BASE_NAME
    assert _value_after(command, "--lora-modules") == (
        f"{config.ADAPTER_NAME}={config.ADAPTER_PATH}"
    )
    assert _value_after(command, "--max-model-len") == str(config.EVAL_MAX_SEQ_LENGTH)


def test_fast_boot_switches_eager_mode_on_and_off():
    assert "--enforce-eager" in modal_server.vllm_command(fast_boot=True)
    assert "--enforce-eager" not in modal_server.vllm_command(fast_boot=False)


def test_adapter_path_sits_under_a_mounted_volume():
    mount = config.ADAPTER_PATH.rsplit("/", 1)[0]
    assert mount in modal_server.volumes
