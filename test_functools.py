from __future__ import annotations

import copy
import functools
import itertools
import os
import platform
import random
import time
from typing import Literal, TypeVar
from unittest import mock

import pytest
from jaraco.classes import properties

from jaraco.functools import (
    Throttler,
    assign_params,
    method_cache,
    retry,
    retry_call,
)

_T = TypeVar("_T")


class TestAssignParams:
    def test_positional_only(self) -> None:
        def func(x: int, /, y: int = 2) -> tuple[int, int]:
            return x, y

        assigned = assign_params(func, {"x": 3, "ignored": 4})
        assert isinstance(assigned, functools.partial)
        assert assigned.func is func
        assert assigned() == (3, 2)

    def test_optional_gap(self) -> None:
        def func(x: int = 1, y: int = 2, /, z: int = 3) -> tuple[int, int, int]:
            return x, y, z

        assigned = assign_params(func, {"y": 4})
        assert assigned() == (1, 4, 3)
        assert assigned(5) == (1, 4, 5)

    def test_optional_suffix_remains_open(self) -> None:
        def func(x: int = 1, y: int = 2, /) -> tuple[int, int]:
            return x, y

        assigned = assign_params(func, {"x": 3})
        assert assigned() == (3, 2)
        assert assigned(4) == (3, 4)

    def test_required_suffix_remains_open(self) -> None:
        def func(x: int, y: int, /) -> tuple[int, int]:
            return x, y

        assigned = assign_params(func, {"x": 3})
        with pytest.raises(TypeError, match='argument'):
            assigned()
        assert assigned(4) == (3, 4)

    def test_required_gap(self) -> None:
        def func(x: int, y: int, /) -> tuple[int, int]:
            return x, y

        with pytest.raises(TypeError, match="missing required parameter 'x'"):
            assign_params(func, {"y": 4})
        assert assign_params(func, {"x": 3, "y": 4})() == (3, 4)

    def test_required_gap_with_kwargs(self) -> None:
        def func(x: int, y: int, /, **kwargs: int) -> tuple[int, int, dict[str, int]]:
            return x, y, kwargs

        with pytest.raises(TypeError, match="missing required parameter 'x'"):
            assign_params(func, {"y": 4})
        assert assign_params(func, {"x": 3, "y": 4})(y=5) == (3, 4, {"y": 5})

    def test_unassigned_positional_only_and_keyword_overrides(self) -> None:
        def func(x: int, /, y: int = 2, *, z: int = 3) -> tuple[int, int, int]:
            return x, y, z

        assigned = assign_params(func, {"y": 4, "z": 5})
        assert assigned(1) == (1, 4, 5)
        assert assigned(1, y=6, z=7) == (1, 6, 7)

    def test_existing_partial(self) -> None:
        def func(x: int = 1, y: int = 2, z: int = 3, /) -> tuple[int, int, int]:
            return x, y, z

        assigned = assign_params(functools.partial(func, 4), {"x": 9, "z": 5})
        assert assigned() == (4, 2, 5)

    def test_bound_method(self) -> None:
        class Handler:
            def method(self, arg: int, /, other: int = 2) -> tuple[int, int]:
                return arg, other

        assert assign_params(Handler().method, {"arg": 3})() == (3, 2)

    def test_builtin(self) -> None:
        assert assign_params(divmod, {"x": 7, "y": 3})() == (2, 1)

    def test_keyword_with_positional_only_name(self) -> None:
        def func(x: int, /, **kwargs: int) -> tuple[int, dict[str, int]]:
            return x, kwargs

        assigned = assign_params(func, {"x": 3})
        assert assigned() == (3, {})
        assert assigned(x=4) == (3, {"x": 4})


class TestThrottler:
    @pytest.mark.xfail(
        'GITHUB_ACTIONS' in os.environ and platform.system() in ('Darwin', 'Windows'),
        reason="Performance is heavily throttled on Github Actions Mac/Windows runs",
    )
    def test_function_throttled(self) -> None:
        """
        Ensure the throttler actually throttles calls.
        """
        # set up a function to be called
        counter = itertools.count()
        # set up a version of `next` that is only called 30 times per second
        limited_next = Throttler(next, 30)
        # for one second, call next as fast as possible
        deadline = time.time() + 1
        while time.time() < deadline:
            limited_next(counter)
        # ensure the counter was advanced about 30 times
        assert 28 <= next(counter) <= 32

        # ensure that another burst of calls after some idle period will also
        # get throttled
        time.sleep(1)
        deadline = time.time() + 1
        counter = itertools.count()
        while time.time() < deadline:
            limited_next(counter)
        assert 28 <= next(counter) <= 32

    def test_reconstruct_unwraps(self) -> None:
        """
        The throttler should be re-usable - if one wants to throttle a
        function that's aready throttled, the original function should be
        used.
        """
        wrapped = Throttler(next, 30)
        wrapped_again = Throttler(wrapped, 60)
        assert wrapped_again.func is next
        assert wrapped_again.max_rate == 60

    def test_throttled_method(self) -> None:
        class ThrottledMethodClass:
            @Throttler
            def echo(self, arg: _T) -> _T:
                return arg

        tmc = ThrottledMethodClass()
        assert tmc.echo('foo') == 'foo'


class TestMethodCache:
    bad_vers = '(3, 5, 0) <= sys.version_info < (3, 5, 2)'

    @pytest.mark.skipif(bad_vers, reason="https://bugs.python.org/issue25447")
    def test_deepcopy(self) -> None:
        """
        A deepcopy of an object with a method cache should still
        succeed.
        """

        class ClassUnderTest:
            calls = 0

            @method_cache
            def method(self, value: _T) -> _T:
                self.calls += 1
                return value

        ob = ClassUnderTest()
        copy.deepcopy(ob)
        ob.method(1)
        copy.deepcopy(ob)

    def test_special_methods(self) -> None:
        """
        Test method_cache with __getitem__ and __getattr__.
        """

        class ClassUnderTest:
            getitem_calls = 0
            getattr_calls = 0

            @method_cache
            def __getitem__(self, item: _T) -> _T:
                self.getitem_calls += 1
                return item

            @method_cache
            def __getattr__(self, name: _T) -> _T:
                self.getattr_calls += 1
                return name

        ob = ClassUnderTest()

        # __getitem__
        ob[1] + ob[1]
        assert ob.getitem_calls == 1

        # __getattr__
        ob.one + ob.one  # type: ignore[operator] # Using ParamSpec on methods is still limited
        assert ob.getattr_calls == 1

    @pytest.mark.xfail(reason="can't replace property with cache; #6")
    def test_property(self) -> None:
        """
        Can a method_cache decorated method also be a property?
        """

        class ClassUnderTest:
            @property
            @method_cache
            def mything(self) -> float:  # pragma: nocover
                return random.random()

        ob = ClassUnderTest()

        assert ob.mything == ob.mything

    @pytest.mark.xfail(reason="can't replace property with cache; #6")
    def test_non_data_property(self) -> None:
        """
        A non-data property also does not work because the property
        gets replaced with a method.
        """

        class ClassUnderTest:
            @properties.NonDataProperty
            @method_cache
            def mything(self) -> float:
                return random.random()

        ob = ClassUnderTest()

        assert ob.mything == ob.mything


class TestRetry:
    def attempt(self, arg: mock.Mock | None = None) -> Literal['Success']:
        if next(self.fails_left):
            raise ValueError("Failed!")
        if arg:
            arg.touch()
        return "Success"

    def set_to_fail(self, times: int) -> None:
        self.fails_left = itertools.count(times, -1)

    def test_set_to_fail(self) -> None:
        """
        Test this test's internal failure mechanism.
        """
        self.set_to_fail(times=2)
        with pytest.raises(ValueError):
            self.attempt()
        with pytest.raises(ValueError):
            self.attempt()
        assert self.attempt() == 'Success'

    def test_retry_call_succeeds(self) -> None:
        self.set_to_fail(times=2)
        res = retry_call(self.attempt, retries=2, trap=ValueError)
        assert res == "Success"

    def test_retry_call_fails(self) -> None:
        """
        Failing more than the number of retries should
        raise the underlying error.
        """
        self.set_to_fail(times=3)
        with pytest.raises(ValueError) as res:
            retry_call(self.attempt, retries=2, trap=ValueError)
        assert str(res.value) == 'Failed!'

    def test_retry_multiple_exceptions(self) -> None:
        self.set_to_fail(times=2)
        errors = ValueError, NameError
        res = retry_call(self.attempt, retries=2, trap=errors)
        assert res == "Success"

    def test_retry_exception_superclass(self) -> None:
        self.set_to_fail(times=2)
        res = retry_call(self.attempt, retries=2, trap=Exception)
        assert res == "Success"

    def test_default_traps_nothing(self) -> None:
        self.set_to_fail(times=1)
        with pytest.raises(ValueError):
            retry_call(self.attempt, retries=1)

    def test_default_does_not_retry(self) -> None:
        self.set_to_fail(times=1)
        with pytest.raises(ValueError):
            retry_call(self.attempt, trap=Exception)

    def test_cleanup_called_on_exception(self) -> None:
        calls = random.randint(1, 10)
        cleanup = mock.Mock()
        self.set_to_fail(times=calls)
        retry_call(self.attempt, retries=calls, cleanup=cleanup, trap=Exception)
        assert cleanup.call_count == calls
        cleanup.assert_called_with()

    def test_infinite_retries(self) -> None:
        self.set_to_fail(times=999)
        cleanup = mock.Mock()
        retry_call(self.attempt, retries=float('inf'), cleanup=cleanup, trap=Exception)
        assert cleanup.call_count == 999

    def test_with_arg(self) -> None:
        self.set_to_fail(times=0)
        arg = mock.Mock()
        bound = functools.partial(self.attempt, arg)
        res = retry_call(bound)
        assert res == 'Success'
        assert arg.touch.called

    def test_decorator(self) -> None:
        self.set_to_fail(times=1)
        attempt = retry(retries=1, trap=Exception)(self.attempt)
        res = attempt()
        assert res == "Success"

    def test_decorator_with_arg(self) -> None:
        self.set_to_fail(times=0)
        attempt = retry()(self.attempt)
        arg = mock.Mock()
        res = attempt(arg)
        assert res == 'Success'
        assert arg.touch.called
