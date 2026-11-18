# Building the ygopro-core oracle (libocgcore)

`setup.sh` does all of this on Linux and macOS, with ygopro-core pinned to the commit validated against the JAX
engine. This note records what the build needs and what the oracle exposes.

## Dependencies

- **Lua 5.4**: not installed separately. It is vendored as a git submodule in `lua/src` and built with the project.
- **premake5**: `brew install premake` on macOS; `setup.sh` downloads a release binary into `.tools/` on Linux.
- **Compiler**: a C++17 compiler (clang++ or g++).
- **Python 3**: only the standard `ctypes` module.

## Commands

```sh
cd oracle   # from the repository root
git clone --depth 1 --recursive https://github.com/edo9300/ygopro-core ygopro-core

cd ygopro-core
premake5 gmake2
make -C build ocgcoreshared config=release -j8

cd ../..
python3 oracle/smoke.py
```

`make -C build ocgcoreshared config=release` produces `bin/release/libocgcore.so` (`.dylib` on macOS). The `ocgcore`
target would produce a static library instead. Symbol export is enabled by the `OCGCORE_EXPORT_FUNCTIONS`
definition, which premake applies to the `ocgcoreshared` target.

## Smoke test

`smoke.py` loads the library from a path relative to its own location and calls
`OCG_GetVersion(int* major, int* minor)`. Expected output:

```
OCG_GetVersion -> 11.0
```

## Exported functions (ocgapi.h)

The 13 exported symbols, checked with `nm`:

```c
/*** CORE INFORMATION ***/
void OCG_GetVersion(int* major, int* minor);

/*** DUEL CREATION AND DESTRUCTION ***/
int OCG_CreateDuel(OCG_Duel* out_ocg_duel, const OCG_DuelOptions* options_ptr);
void OCG_DestroyDuel(OCG_Duel ocg_duel);
void OCG_DuelNewCard(OCG_Duel ocg_duel, const OCG_NewCardInfo* info_ptr);
void OCG_StartDuel(OCG_Duel ocg_duel);

/*** DUEL PROCESSING AND QUERYING ***/
int OCG_DuelProcess(OCG_Duel ocg_duel);
void* OCG_DuelGetMessage(OCG_Duel ocg_duel, uint32_t* length);
void OCG_DuelSetResponse(OCG_Duel ocg_duel, const void* buffer, uint32_t length);
int OCG_LoadScript(OCG_Duel ocg_duel, const char* buffer, uint32_t length, const char* name);

uint32_t OCG_DuelQueryCount(OCG_Duel ocg_duel, uint8_t team, uint32_t loc);
void* OCG_DuelQuery(OCG_Duel ocg_duel, uint32_t* length, const OCG_QueryInfo* info_ptr);
void* OCG_DuelQueryLocation(OCG_Duel ocg_duel, uint32_t* length, const OCG_QueryInfo* info_ptr);
void* OCG_DuelQueryField(OCG_Duel ocg_duel, uint32_t* length);
```

The types (`OCG_Duel`, `OCG_DuelOptions`, ...) are defined in `ocgapi_types.h`.

`build_inspector.py` also compiles `effect_inspector.cpp` against the oracle's headers: a small read-only extension
that exposes card identities and Xyz materials, so that `compare.py` can check the identity of each card copy.
