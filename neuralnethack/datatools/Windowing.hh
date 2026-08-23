#ifndef __Windowing_hh__
#define __Windowing_hh__

#include "CoreDataSet.hh"
#include "DataSet.hh"

#include <memory>

namespace DataTools {

/**Expand a raw weekly table into the channel-major lag-window layout
 * the Adstock stage consumes.
 *
 * The raw DataSet must be in chronological order, one row per period,
 * with inputs laid out as [channel_0 .. channel_{C-1}, covariate_0 ..
 * covariate_{P-1}] and the target(s) as outputs. The result has one
 * pattern per period t >= lags-1 with inputs
 *   [c0 lag0..lagL-1, c1 lag0..lagL-1, ..., covariates of period t]
 * (lag 0 = period t, older lags after) and period t's outputs. The
 * first lags-1 rows serve as warmup history only; their targets are
 * not used.
 *
 * This is the missing step between "a traditional MMM CSV" and this
 * library: with it, a raw table goes straight into
 * `Adstock(channels, lags, passthrough, ...)`.
 *
 * \param raw the chronological raw table (see layout above).
 * \param channels number of media channels C.
 * \param lags window length L.
 * \param passthrough trailing covariate columns copied per period.
 * \return a new CoreDataSet with raw.size() - lags + 1 windowed rows.
 */
std::shared_ptr<CoreDataSet> windowLagged(DataSet& raw, uint channels, uint lags, uint passthrough);

} // namespace DataTools

#endif
