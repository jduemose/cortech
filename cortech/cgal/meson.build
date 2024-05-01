cgal_dir = 'cortech' / 'cgal'

python_sources = [
    '__init__.py',
]
python.install_sources(python_sources, subdir: cgal_dir)

cpp_cgal_args = cpp.get_supported_arguments(
    '-Os',              # optimize for size
    '-flto=auto',       # apply standard link-time optimizer
    '-frounding-math',  # Disable optimizations related to float rounding
)

cpp_cgal_macros = [
    '-DCGAL_MESH_3_NO_DEPRECATED_SURFACE_INDEX',
    '-DCGAL_MESH_3_NO_DEPRECATED_C3T3_ITERATORS',
    '-DCGAL_CONCURRENT_MESH_3',
    '-DCGAL_EIGEN3_ENABLED',
    '-DCGAL_USE_ZLIB',
    '-DCGAL_LINKED_WITH_TBB',
]

cgal_helpers_dep = declare_dependency(
    # include_directories: include_directories('.'), # the current dir is included by default
    sources: 'cgal_helpers.cpp', # the current dir is searched by default
)

pmp = python.extension_module(
    'polygon_mesh_processing',
    cython_gen_cpp.process('polygon_mesh_processing.pyx'),
    cpp_args: [cython_cpp_args, cpp_cgal_args, cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
    install: true,
    subdir: cgal_dir,
)

pmp = python.extension_module(
    'convex_hull_3',
    cython_gen_cpp.process('convex_hull_3.pyx'),
    cpp_args: [cython_cpp_args, cpp_cgal_args, cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
    install: true,
    subdir: cgal_dir,
)