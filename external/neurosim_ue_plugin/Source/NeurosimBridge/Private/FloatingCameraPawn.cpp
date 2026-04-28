// Copyright Neurosim contributors. Licensed under the repo LICENSE.

#include "FloatingCameraPawn.h"

#include "Camera/CameraComponent.h"
#include "Components/SceneComponent.h"
#include "EngineUtils.h"
#include "Engine/World.h"

namespace
{
    // Habitat -> Unreal:
    //   Habitat: y-up, meters, right-handed, quaternion [w,x,y,z].
    //   Unreal : z-up, cm, left-handed, FQuat(x,y,z,w).
    // Mapping used (matches the typical Habitat <-> Unreal convention):
    //   UE.x =  H.x * 100
    //   UE.y = -H.z * 100     (flip z to match handedness; choose z as forward)
    //   UE.z =  H.y * 100
    // The same permutation + sign flip applies to the quaternion imaginary part.
    // This is the simplest convention that preserves "y-up" as UE's "z-up" and
    // negates one axis for handedness. If downstream code expects a different
    // mapping it is localized here.

    constexpr float kMetersToCm = 100.0f;

    FVector HabitatPosToUnreal(const FVector& H)
    {
        return FVector(H.X * kMetersToCm, -H.Z * kMetersToCm, H.Y * kMetersToCm);
    }
    FVector UnrealPosToHabitat(const FVector& U)
    {
        return FVector(U.X / kMetersToCm, U.Z / kMetersToCm, -U.Y / kMetersToCm);
    }
    FQuat HabitatQuatToUnreal(const FQuat& H)
    {
        // Apply same permutation + sign flip to the vector part.
        return FQuat(H.X, -H.Z, H.Y, H.W);
    }
    FQuat UnrealQuatToHabitat(const FQuat& U)
    {
        return FQuat(U.X, U.Z, -U.Y, U.W);
    }
}

AFloatingCameraPawn::AFloatingCameraPawn()
{
    PrimaryActorTick.bCanEverTick = false;

    USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
    RootComponent = Root;

    Camera = CreateDefaultSubobject<UCameraComponent>(TEXT("Camera"));
    Camera->SetupAttachment(Root);
}

AFloatingCameraPawn* AFloatingCameraPawn::GetOrSpawn(UWorld* World)
{
    if (!World) return nullptr;
    for (TActorIterator<AFloatingCameraPawn> It(World); It; ++It) return *It;

    FActorSpawnParameters Params;
    Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    Params.Name = TEXT("NeurosimFloatingCameraPawn");
    return World->SpawnActor<AFloatingCameraPawn>(
        AFloatingCameraPawn::StaticClass(),
        FVector::ZeroVector, FRotator::ZeroRotator, Params);
}

void AFloatingCameraPawn::SetHabitatPose(const FVector& HabitatPosition, const FQuat& HabitatRotation)
{
    const FVector UePos = HabitatPosToUnreal(HabitatPosition);
    const FQuat UeRot = HabitatQuatToUnreal(HabitatRotation);
    SetActorLocationAndRotation(UePos, UeRot, /*bSweep=*/false, nullptr,
                                ETeleportType::TeleportPhysics);
}

void AFloatingCameraPawn::GetHabitatPose(FVector& OutHabitatPosition, FQuat& OutHabitatRotation) const
{
    OutHabitatPosition = UnrealPosToHabitat(GetActorLocation());
    OutHabitatRotation = UnrealQuatToHabitat(GetActorQuat());
}
