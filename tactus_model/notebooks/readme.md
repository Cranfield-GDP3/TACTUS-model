This directory keeps the project notebooks. Tracking notebooks
using git is sub optimal because each time a cell is executed
it changes output, which creates an unstaged file (a file with
new modifications). This could lead to the file being present in
every commit and being impossible to track down.
