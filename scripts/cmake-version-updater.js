// standard-version custom updater: keeps the CMake project() VERSION in sync
// with the released version. Referenced from .versionrc -> bumpFiles.
const RE = /(project\([^\)]*?VERSION\s+)(\d+\.\d+\.\d+)/;

module.exports.readVersion = function (contents) {
  const m = contents.match(RE);
  return m ? m[2] : undefined;
};

module.exports.writeVersion = function (contents, version) {
  return contents.replace(RE, `$1${version}`);
};
