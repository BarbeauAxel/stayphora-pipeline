"""Les étapes du pipeline, dans l'ordre. Chaque module expose :
NAME, run(ctx) -> métriques, verify(ctx, métriques), outputs(ctx) -> chemins attendus.
"""

from . import (s1_fetch, s2_frames, s3_colmap, s4_train, s5_export, s6_publish, s7_tour_entry,
               s8_notify)

STEPS = [s1_fetch, s2_frames, s3_colmap, s4_train, s5_export, s6_publish, s7_tour_entry, s8_notify]
