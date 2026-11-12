"""Lightweight row facade for hot per-row scoring loops.

``pandas.DataFrame.iterrows()`` materializes a ``pd.Series`` per row —
the dominant cost in the batch scoring paths. ``RowView`` exposes the
small subset of the Series API the engines actually use (``.get``,
``.index`` membership, ``[]``, ``in``, ``.copy``) over a plain dict
produced by ``DataFrame.to_dict('records')``, which is roughly an
order of magnitude cheaper per row.

Only read access is supported; ``copy()`` returns a mutable RowView so
callers can attach derived fields (e.g. ``risk_factors``) exactly like
they did on a Series.
"""

from __future__ import annotations

from typing import Any, Dict


class RowView:
    """Read-mostly dict facade mimicking the pd.Series accessors used
    by the detection/risk engines."""

    __slots__ = ('_d',)

    def __init__(self, data: Dict[str, Any]):
        self._d = data

    @property
    def index(self):
        # dict.keys() supports ``in`` membership tests, matching the
        # ``'col' in row.index`` idiom used throughout the engines.
        return self._d.keys()

    @property
    def name(self):
        return self._d.get('_row_name')

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._d[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def copy(self) -> 'RowView':
        return RowView(dict(self._d))

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._d)
