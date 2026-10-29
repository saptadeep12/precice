#include <mpi.h>
#include <precice/precice.hpp>

#include <cstdlib>
#include <memory>
#include <string>

namespace {

int getAppnum()
{
  int *appnum = nullptr;
  int  flag    = 0;
  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_APPNUM, &appnum, &flag);
  return flag && appnum != nullptr ? *appnum : -1;
}

int checkCommunicator(
    const char *configuration,
    const char *participantName,
    int         solverProcessIndex,
    int         solverProcessSize,
    MPI_Comm   *communicator,
    bool        useCommunicator,
    MPI_Comm    expectedCommunicator,
    int         expectedComparison)
{
  std::unique_ptr<precice::Participant> participant;
  if (useCommunicator) {
    participant = std::make_unique<precice::Participant>(participantName,
                                                         configuration,
                                                         solverProcessIndex,
                                                         solverProcessSize,
                                                         communicator);
  } else {
    participant = std::make_unique<precice::Participant>(participantName,
                                                         configuration,
                                                         solverProcessIndex,
                                                         solverProcessSize);
  }
  auto *actual = static_cast<MPI_Comm *>(participant->getCommunicator());

  int actualRank = -1;
  int actualSize = -1;
  MPI_Comm_rank(*actual, &actualRank);
  MPI_Comm_size(*actual, &actualSize);

  int comparison = MPI_UNEQUAL;
  MPI_Comm_compare(*actual, expectedCommunicator, &comparison);
  return actualRank == solverProcessIndex && actualSize == solverProcessSize && comparison == expectedComparison
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

} // namespace

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  if (argc != 3) {
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  int worldRank = -1;
  int worldSize = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  const std::string mode = argv[1];
  const int         appnum = getAppnum();
  if (appnum < 0 && mode == "custom" && worldSize == 5) {
    MPI_Finalize();
    return 77;
  }

  int minimumAppnum = appnum;
  int maximumAppnum = appnum;
  MPI_Allreduce(&appnum, &minimumAppnum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&appnum, &maximumAppnum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if ((minimumAppnum < 0 || minimumAppnum == maximumAppnum) && worldSize == 5 && mode != "ordinary") {
    MPI_Finalize();
    return 77;
  }

  MPI_Comm expected      = MPI_COMM_WORLD;
  bool     expectedOwned = false;
  if (appnum >= 0) {
    MPI_Comm_split(MPI_COMM_WORLD, appnum, worldRank, &expected);
    expectedOwned = true;
  }

  int expectedRank = -1;
  int expectedSize = -1;
  MPI_Comm_rank(expected, &expectedRank);
  MPI_Comm_size(expected, &expectedSize);

  const char *name = appnum == 0 ? "SolverOne" : "SolverTwo";
  int         result;

  if (mode == "automatic") {
    result = checkCommunicator(argv[2], name, 0, 1, nullptr, false, expected, appnum < 0 ? MPI_IDENT : MPI_CONGRUENT);
  } else if (mode == "ordinary") {
    result = checkCommunicator(argv[2], name, worldRank, worldSize, nullptr, false, MPI_COMM_WORLD, MPI_IDENT);
    if (result == EXIT_SUCCESS) {
      result = checkCommunicator(argv[2], name, worldRank, worldSize, nullptr, false, MPI_COMM_WORLD, MPI_IDENT);
    }
  } else if (mode == "custom") {
    MPI_Comm custom = MPI_COMM_WORLD;
    result         = checkCommunicator(argv[2], name, worldRank, worldSize, &custom, true, MPI_COMM_WORLD, MPI_IDENT);
  } else if (mode == "custom-wrong") {
    MPI_Comm custom = MPI_COMM_WORLD;
    bool      threw  = false;
    try {
      checkCommunicator(argv[2], "SolverOne", worldRank == 0 ? 1 : 0, worldSize, &custom, true, MPI_COMM_WORLD, MPI_IDENT);
    } catch (const precice::Error &) {
      threw = true;
    }
    bool allRanksThrew = false;
    MPI_Allreduce(&threw, &allRanksThrew, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    result = allRanksThrew ? EXIT_SUCCESS : EXIT_FAILURE;
  } else {
    result = EXIT_FAILURE;
  }

  if (expectedOwned) {
    MPI_Comm_free(&expected);
  }
  MPI_Finalize();
  return result;
}
