def get_step_count(B,T):
    TOTAL_BATCH_SIZE = 524288 # Roughly 0.5M tokens as per the gpt3 paper
    assert TOTAL_BATCH_SIZE % (B * T) == 0
    accum_steps = TOTAL_BATCH_SIZE//(B*T)
    print(f" Steps per iteration: {accum_steps}")
    return accum_steps


