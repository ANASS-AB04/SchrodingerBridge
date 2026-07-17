import logging
import pickle
import tempfile
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Iterator, Type, cast

from phdtruel.fields.parameters import ParameterSet

logger = logging.getLogger(__name__)


class CachableItem(metaclass=ABCMeta):
    @property
    @abstractmethod
    def parameter(self) -> ParameterSet:
        raise NotImplementedError

    @property
    @abstractmethod
    def cache_dir(self) -> tempfile.TemporaryDirectory:
        raise NotImplementedError

    @property
    @abstractmethod
    def nbytes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other: "CachableItem") -> "CachableItem":
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other: "CachableItem") -> "CachableItem":
        raise NotImplementedError


class CachableOrderedDict:
    """
    A class that supports iteration and addition, and caches its elements as temporary files.
    """

    def __init__(
        self,
        *args: CachableItem,
        datatype: Type[CachableItem] | None = None,
        max_memory: int = 1_000_000_000,
    ):
        self.cache_dir = tempfile.TemporaryDirectory(
            # dir="/home/matruel/Documents/tmp_folder"
        )

        if datatype is None and len(args) > 1:
            datatype = type(args[0])

        self._values: list[CachableItem | Path] = list(args)
        self._parameters: list[ParameterSet] = [f.parameter for f in args]
        self._datatype = datatype
        self._caching_queue = list(range(len(args)))
        self._nbytes_sizes = [f.nbytes for f in args]
        self._metadata = set()

        self.max_memory = max_memory

        self._update_cache()

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[CachableItem]:
        for k in range(len(self._values)):
            yield self[k]

    def __getitem__(self, index: int | ParameterSet) -> CachableItem:
        if isinstance(index, ParameterSet):
            index = self._parameters.index(index)

        value = self._values[index]
        if isinstance(value, Path):
            self.to_memory(index)
        value = self._values[index]
        self._update_cache()
        return cast(CachableItem, value)

    def items(self) -> dict[ParameterSet, CachableItem]:
        return dict(zip(self._parameters, self.values()))

    def keys(self) -> list[ParameterSet]:
        return self.parameters

    @property
    def parameters(self) -> list[ParameterSet]:
        return self._parameters

    def values(self) -> Iterator[CachableItem]:
        for k in range(len(self._values)):
            yield self[k]

    def __or__(
        self, other: dict[ParameterSet, CachableItem] | "CachableOrderedDict"
    ) -> dict[ParameterSet, CachableItem] | "CachableOrderedDict":
        if isinstance(other, dict):
            return {**self.items()} | other
        elif isinstance(other, CachableOrderedDict):
            raise NotImplementedError
        else:
            raise ValueError("Can only operate on dicts or CachableFieldsOrderedDicts")

    def __add__(self, other: list[CachableItem] | "CachableOrderedDict"):
        if isinstance(other, list):
            other = CachableOrderedDict(*other, max_memory=self.max_memory)
        if not isinstance(other, CachableOrderedDict):
            raise ValueError("Only CacheableSequence instances can be added")
        new = CachableOrderedDict(max_memory=min(self.max_memory, self.max_memory))
        for value in other:
            new.append(value)
        for value in self:
            new.append(value)
        return new

    def append(self, field: CachableItem) -> None:
        if self._datatype is None:
            self._datatype = type(field)
        if not isinstance(field, self._datatype):
            raise ValueError("type of value is incompatible with other values")

        self._values.append(field)
        self._parameters.append(field.parameter)
        new_index = len(self._values) - 1
        self._caching_queue.append(new_index)
        self._nbytes_sizes.append(field.nbytes)

        self._update_cache()

    def nbytes(self) -> int:
        return sum(self._nbytes_sizes)

    def _update_cache(self) -> None:
        # num_to_cache = max(0, len(self) - self.max_in_memory)
        # for index in self._caching_queue[:num_to_cache]:
        #     self.to_cache(index)

        cumulative_memory = 0
        for index in self._caching_queue[::-1]:
            cumulative_memory += self._nbytes_sizes[index]
            if cumulative_memory > self.max_memory:
                self.to_cache(index)
        # logger.debug(f"Memory {_sizeof_fmt(cumulative_memory)}")

    def to_cache(self, index: int) -> None:
        value = self._values[index]
        if isinstance(value, Path):
            return
        cache_file_path = Path(self.cache_dir.name) / f"index_{index}.pickle"
        logger.debug(
            f"Caching {index} to {cache_file_path} in {id(self)} that tracks {self._metadata}"
        )

        try:
            # Prevent the deletion of cachedir for the value
            tmp_dir = value.cache_dir
            logger.debug(f"saving in meta {tmp_dir}")
            if tmp_dir not in self._metadata:
                self._metadata.add(tmp_dir)
        except AttributeError:
            pass
        with open(cache_file_path, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._values[index] = cache_file_path
        del value

    def to_memory(self, index: int) -> None:
        value = self._values[index]
        if isinstance(value, CachableItem):
            return
        logger.debug(
            f"loading {index} from {value} in {id(self)} that tracks {self._metadata}"
        )
        with open(value, "rb") as handle:
            self._values[index] = pickle.load(handle)
        self._set_last_in_cache_queue(index)  # Because it was just accessed
        # os.remove(value) # TODO : Free the cache as is being loaded to memory
        del value

    def _set_last_in_cache_queue(self, index: int) -> None:
        current_place = self._caching_queue.index(index)
        del self._caching_queue[current_place]
        self._caching_queue.append(index)

    def __del__(self) -> None:
        logger.debug(f"deleting {self} with {self.cache_dir}")
        # Path(self.cache_dir.name).exists()
