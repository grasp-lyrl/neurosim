// Copyright Neurosim contributors. Licensed under the repo LICENSE.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "FloatingCameraPawn.generated.h"

class UCameraComponent;

/**
 * Disembodied floating camera, analogous to Habitat's default agent. No physics,
 * no collisions; transform is driven directly by the Neurosim bridge.
 *
 * Coordinate conversion: Python speaks Habitat (y-up, meters, quaternion [w,x,y,z]).
 * Unreal is z-up, cm, left-handed. Conversion lives here, not on the Python side.
 */
UCLASS()
class AFloatingCameraPawn : public APawn
{
    GENERATED_BODY()

public:
    AFloatingCameraPawn();

    /** Find an existing pawn in World or spawn one. Game thread only. */
    static AFloatingCameraPawn* GetOrSpawn(UWorld* World);

    /** Set pose from Habitat-frame inputs. */
    void SetHabitatPose(const FVector& HabitatPosition, const FQuat& HabitatRotation);

    /** Read current pose into Habitat-frame outputs. */
    void GetHabitatPose(FVector& OutHabitatPosition, FQuat& OutHabitatRotation) const;

private:
    UPROPERTY(VisibleAnywhere)
    TObjectPtr<UCameraComponent> Camera;
};
