using Hybridizer.Runtime.CUDAImports;
using System;

namespace SimpleMetadataDecorator
{
	public struct ThreadLocalDictionary<TKey, TValue>
	{
		private readonly TKey[] _keys;
		private readonly TValue[] _values;
		private int _capacity;
		private int _nextInsertIndex;

		public ThreadLocalDictionary(int capacity)
		{
			_keys = new TKey[capacity];
			_values = new TValue[capacity];
			_capacity = capacity;
			_nextInsertIndex = 0;
		}

		public TKey[] Keys { get { return _keys; } }
		public TValue[] Values { get { return _values; } }

		[Kernel]
		public int Count
		{
			get
			{
				return _nextInsertIndex;
			}
		}

		[Kernel]
		public bool TryGetValue(TKey key, out TValue val)
		{

			int k = 0;
			while (k < _nextInsertIndex)
			{
				if (object.Equals(_keys[k], key))
				{
					val = _values[k];
					return true;
				}
				++k;
			}

			val = default(TValue);
			return false;
		}

		[Kernel]
		public TValue this[TKey key]
		{
			get
			{
				int k = 0;
				while (k < _nextInsertIndex)
				{
					if (_keys[k].Equals(key))
					{
						return _values[k];
					}
					++k;
				}

				return default(TValue);
			}
		}

		[Kernel]
		public bool Remove(TKey key)
		{
			int k = 0;
			bool found = false;
			while (k < _nextInsertIndex)
			{
				if (_keys[k].Equals(key))
				{
					found = true;
					break;
				}

				++k;
			}

			if (found)
			{
				while (k < _nextInsertIndex && k < _capacity - 1)
				{
					_keys[k] = _keys[k + 1];
					++k;
				}

				_nextInsertIndex -= 1;
			}

			return found;
		}

		// TODO: device-realloc
		[HybridizerIgnore]
		public void Add(TKey key, TValue val)
		{
			if (_nextInsertIndex == _capacity)
				throw new OutOfMemoryException();
			_keys[_nextInsertIndex] = key;
			_values[_nextInsertIndex] = val;
			_nextInsertIndex += 1;
		}
	}
}
