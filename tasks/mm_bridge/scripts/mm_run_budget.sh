#!/bin/bash

MM_TRAIN_SAMPLE_BUDGET="${MM_TRAIN_SAMPLE_BUDGET:-1152000}"

mm_effective_batch_size() {
  local batch_size="$1"
  local grad_accum_steps="${2:-1}"
  echo $(( batch_size * grad_accum_steps ))
}

mm_budget_steps_for_bs_ga() {
  local batch_size="$1"
  local grad_accum_steps="${2:-1}"
  local eff_batch
  eff_batch="$(mm_effective_batch_size "${batch_size}" "${grad_accum_steps}")"
  echo $(( (MM_TRAIN_SAMPLE_BUDGET + eff_batch - 1) / eff_batch ))
}
