#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:58:05 2022

@author: gus
"""
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _get_observation_pairs, _split_observation_pairs, _build_observation_dict, _calculate_weights_below_for_multi_objective
from typing import Any
from typing import Dict
from typing import Union

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.study import Study
from optuna.trial import FrozenTrial

class CustomTPESampler(TPESampler):
    def predict_tpe_scores(
            self,
            study: Study,
            param_name: str,
            param_distribution: BaseDistribution,
            samples: np.ndarray,
    ) -> Any:
        
        values, scores = _get_observation_pairs(
            study, [param_name], self._multivariate, self._constant_liar
        )
        
        n = len(scores)

        indices_below, indices_above = _split_observation_pairs(scores, self._gamma(n))
        # `None` items are intentionally converted to `nan` and then filtered out.
        # For `nan` conversion, the dtype must be float.
        config_values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        below = _build_observation_dict(config_values, indices_below)
        above = _build_observation_dict(config_values, indices_above)

        if study._is_multi_objective():
            weights_below = _calculate_weights_below_for_multi_objective(
                config_values, scores, indices_below
            )
            mpe_below = _ParzenEstimator(
                below,
                {param_name: param_distribution},
                self._parzen_estimator_parameters,
                weights_below,
            )
        else:
            mpe_below = _ParzenEstimator(
                below, {param_name: param_distribution}, self._parzen_estimator_parameters
            )
        mpe_above = _ParzenEstimator(
            above, {param_name: param_distribution}, self._parzen_estimator_parameters
        )
        samples_below = {param_name: samples}
        log_likelihoods_below = mpe_below.log_pdf(samples_below)
        log_likelihoods_above = mpe_above.log_pdf(samples_below)
        return log_likelihoods_below - log_likelihoods_above
    
    def sample_independent_override(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        
        values, scores = _get_observation_pairs(
            study, [param_name], self._multivariate, self._constant_liar
        )
        
        n = len(scores)

        self._log_independent_sampling(n, trial, param_name)

        if n < self._n_startup_trials:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )
        
        indices_below, indices_above = _split_observation_pairs(scores, self._gamma(n))
        # `None` items are intentionally converted to `nan` and then filtered out.
        # For `nan` conversion, the dtype must be float.
        config_values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        below = _build_observation_dict(config_values, indices_below)
        above = _build_observation_dict(config_values, indices_above)

        if study._is_multi_objective():
            weights_below = _calculate_weights_below_for_multi_objective(
                config_values, scores, indices_below
            )
            mpe_below = _ParzenEstimator(
                below,
                {param_name: param_distribution},
                self._parzen_estimator_parameters,
                weights_below,
            )
        else:
            mpe_below = _ParzenEstimator(
                below, {param_name: param_distribution}, self._parzen_estimator_parameters
            )
        mpe_above = _ParzenEstimator(
            above, {param_name: param_distribution}, self._parzen_estimator_parameters
        )
        samples_below = mpe_below.sample(self._rng, self._n_ei_candidates)
        log_likelihoods_below = mpe_below.log_pdf(samples_below)
        log_likelihoods_above = mpe_above.log_pdf(samples_below)
        ret = CustomTPESampler._compare_override(samples_below, log_likelihoods_below, log_likelihoods_above)
        return {'best_param_val':param_distribution.to_external_repr(ret[param_name]),
                'best_score':ret['best_score'],
                'samples': samples_below[param_name],
                'scores': ret['scores']}

    @classmethod
    def _compare_override(
        cls,
        samples: Dict[str, np.ndarray],
        log_l: np.ndarray,
        log_g: np.ndarray,
    ) -> Dict[str, Union[float, int]]:
        
        sample_size = next(iter(samples.values())).size
        if sample_size:
            score = log_l - log_g
            if sample_size != score.size:
                raise ValueError(
                    "The size of the 'samples' and that of the 'score' "
                    "should be same. "
                    "But (samples.size, score.size) = ({}, {})".format(sample_size, score.size)
                )
            best = np.argmax(score)
            ret = {k: v[best].item() for k, v in samples.items()}
            ret['best_score'] = score[best]
            ret['scores'] = score
            return ret
        else:
            raise ValueError(
                "The size of 'samples' should be more than 0."
                "But samples.size = {}".format(sample_size)
            )