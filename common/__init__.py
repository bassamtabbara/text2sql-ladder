"""Frozen shared harness for text2sql-ladder.

Everything in this package is meant to stay stable across all rungs. If you change how
evaluation works, you break comparability between rungs, which is the one thing that makes
this whole exercise worth doing. Change these files only with a very good reason, and if you
do, re-run every rung.
"""
