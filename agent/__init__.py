"""Generator-Critic multi-agent workflow.

* :mod:`agent.generator` proposes strategy candidates (alpha hypotheses).
* :mod:`agent.critic` attacks them: out-of-sample decay, trade frequency, Lo-corrected
  t-statistics with a multiple-testing (snooping) hurdle.
* :mod:`agent.coordinator` routes the loop and gates live signals through the regime
  filter, the risk engine, and the kill switch.
"""
from __future__ import annotations
