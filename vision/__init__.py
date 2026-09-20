"""System One on a vision-language model.

The text pipeline in `systemone/` and this one share every component that
matters: the block mask, the slot head, the loss and the metrics. Only the
packing differs, because an image has to be turned into tokens first.

That is the whole claim this package exists to test.
"""
