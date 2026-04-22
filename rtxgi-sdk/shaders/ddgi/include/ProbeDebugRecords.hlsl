/*
* Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef DDGI_PROBE_DEBUG_RECORDS_ENABLED
    #define DDGI_PROBE_DEBUG_RECORDS_ENABLED 0
#endif

#ifndef DDGI_PROBE_DEBUG_PASS_COUNT
    #define DDGI_PROBE_DEBUG_PASS_COUNT 3
#endif

#ifndef DDGI_PROBE_DEBUG_RECORDS_PER_PROBE
    #define DDGI_PROBE_DEBUG_RECORDS_PER_PROBE 1
#endif

#ifndef DDGI_PROBE_DEBUG_HEADER_ENTRIES
    #define DDGI_PROBE_DEBUG_HEADER_ENTRIES 0
#endif

#ifndef DDGI_PROBE_DEBUG_MAX_VOLUMES
    #define DDGI_PROBE_DEBUG_MAX_VOLUMES 1
#endif

#ifndef DDGI_PROBE_DEBUG_PASS_INDEX
    #define DDGI_PROBE_DEBUG_PASS_INDEX 0
#endif

#ifndef DDGI_PROBE_DEBUG_BLEND_FLAG_INACTIVE_SKIP
    #define DDGI_PROBE_DEBUG_BLEND_FLAG_INACTIVE_SKIP 1
#endif

#ifndef DDGI_PROBE_DEBUG_BLEND_FLAG_BACKFACE_EARLY_OUT
    #define DDGI_PROBE_DEBUG_BLEND_FLAG_BACKFACE_EARLY_OUT 2
#endif

struct DDGIProbeDebugRecord
{
    uint4 probe_meta;
    float4 trace_probe;
    float4 trace_fixed_ray_dir_kind;
    float4 trace_fixed_ray_data;
    float4 trace_random_ray_dir_kind;
    float4 trace_random_ray_data;
    float4 relocate_stats;
    float4 relocate_offset;
    uint4 classify_stats;
    float4 classify_probe;
    float4 irradiance_center;
    float4 distance_center;
    float4 packed_state;
};

#if DDGI_PROBE_DEBUG_RECORDS_ENABLED && RTXGI_DDGI_BINDLESS_RESOURCES && (RTXGI_BINDLESS_TYPE == RTXGI_BINDLESS_TYPE_DESCRIPTOR_HEAP)

#ifndef DDGI_PROBE_DEBUG_UAV_INDEX
    #error Required define DDGI_PROBE_DEBUG_UAV_INDEX is not defined for DDGI probe debug record writes.
#endif

void DDGIWriteProbeDebugRecord(
    uint probeIndex,
    uint totalProbes,
    uint volumeIndex,
    DDGIVolumeDescGPU volume,
    RWTexture2DArray<float4> RayData,
    RWTexture2DArray<float4> ProbeIrradiance,
    RWTexture2DArray<float4> ProbeDistance,
    RWTexture2DArray<float4> ProbeData)
{
    RWStructuredBuffer<DDGIProbeDebugRecord> ProbeDebugOut = ResourceDescriptorHeap[DDGI_PROBE_DEBUG_UAV_INDEX];

    uint debugBufferEntries = 0;
    uint debugBufferStride = 0;
    ProbeDebugOut.GetDimensions(debugBufferEntries, debugBufferStride);

    if (DDGI_PROBE_DEBUG_MAX_VOLUMES == 0) return;

    uint perVolumeEntries = debugBufferEntries / DDGI_PROBE_DEBUG_MAX_VOLUMES;
    if (perVolumeEntries == 0) return;

    uint maxProbeRecordsPerPass = perVolumeEntries / DDGI_PROBE_DEBUG_PASS_COUNT;
    uint probesToDump = min(totalProbes, maxProbeRecordsPerPass);

    uint volumeSlot = min(volumeIndex, DDGI_PROBE_DEBUG_MAX_VOLUMES - 1);
    uint volumeBase = volumeSlot * perVolumeEntries;

    if (probeIndex >= probesToDump) return;

    int3 probeCoords = DDGIGetProbeCoords((int)probeIndex, volume);
    uint3 probeTexel = DDGIGetProbeTexelCoords((int)probeIndex, volume);

    uint probeNumIrradianceTexels = uint(volume.probeNumIrradianceInteriorTexels + 2);
    uint probeNumDistanceTexels = uint(volume.probeNumDistanceInteriorTexels + 2);

    uint2 irradianceCenter = uint2((probeNumIrradianceTexels / 2), (probeNumIrradianceTexels / 2));
    uint2 distanceCenter = uint2((probeNumDistanceTexels / 2), (probeNumDistanceTexels / 2));

    uint2 irradianceTexel = uint2(probeTexel.x * probeNumIrradianceTexels, probeTexel.y * probeNumIrradianceTexels) + irradianceCenter;
    uint2 distanceTexel = uint2(probeTexel.x * probeNumDistanceTexels, probeTexel.y * probeNumDistanceTexels) + distanceCenter;

    float4 irradiance = ProbeIrradiance[uint3(irradianceTexel, probeTexel.z)];
    float2 distanceMoments = ProbeDistance[uint3(distanceTexel, probeTexel.z)].rg * 2.f;
    float4 probeData = ProbeData[probeTexel];

    float probeState = probeData.a;
    float3 probeOffset = DDGILoadProbeDataOffset(ProbeData, probeTexel, volume);
    float3 probeWorldPos = DDGIGetProbeWorldPosition(probeCoords, volume, ProbeData);

    uint maxFixedRays = min((uint)volume.probeNumRays, (uint)RTXGI_DDGI_NUM_FIXED_RAYS);
    uint fixedRayIndex = 0;
    uint randomRayIndex = (volume.probeNumRays > RTXGI_DDGI_NUM_FIXED_RAYS)
        ? (uint)RTXGI_DDGI_NUM_FIXED_RAYS
        : (maxFixedRays > 0 ? (maxFixedRays - 1) : 0);

    uint3 fixedRayTexel = DDGIGetRayDataTexelCoords((int)fixedRayIndex, (int)probeIndex, volume);
    float3 fixedRayDir = DDGIGetProbeRayDirection((int)fixedRayIndex, volume);
    float fixedRayDistance = DDGILoadProbeRayDistance(RayData, fixedRayTexel, volume);
    float3 fixedRayRadiance = DDGILoadProbeRayRadiance(RayData, fixedRayTexel, volume);

    uint3 randomRayTexel = DDGIGetRayDataTexelCoords((int)randomRayIndex, (int)probeIndex, volume);
    float3 randomRayDir = DDGIGetProbeRayDirection((int)randomRayIndex, volume);
    float randomRayDistance = DDGILoadProbeRayDistance(RayData, randomRayTexel, volume);
    float3 randomRayRadiance = DDGILoadProbeRayRadiance(RayData, randomRayTexel, volume);

    uint firstBlendRay = (volume.probeRelocationEnabled || volume.probeClassificationEnabled)
        ? (uint)RTXGI_DDGI_NUM_FIXED_RAYS
        : 0u;
    uint blendedRayCount = (volume.probeNumRays > (int)firstBlendRay)
        ? ((uint)volume.probeNumRays - firstBlendRay)
        : 0u;
    uint maxBackfaces = (uint)((float)blendedRayCount * volume.probeRandomRayBackfaceThreshold);
    uint blendBackfaces = 0u;
    float blendWeightSum = 0.f;
    bool blendBackfaceEarlyOut = false;

    uint centerTexel = ((uint)volume.probeNumIrradianceInteriorTexels) / 2u;
    float2 centerProbeOctantUV = DDGIGetNormalizedOctahedralCoordinates(
        int2((int)centerTexel, (int)centerTexel),
        volume.probeNumIrradianceInteriorTexels);
    float3 centerProbeDirection = DDGIGetOctahedralDirection(centerProbeOctantUV);

    for (uint rayIndex = firstBlendRay; rayIndex < (uint)volume.probeNumRays; rayIndex++)
    {
        int3 blendRayTexel = DDGIGetRayDataTexelCoords((int)rayIndex, (int)probeIndex, volume);
        float blendDistance = DDGILoadProbeRayDistance(RayData, blendRayTexel, volume);
        if (blendDistance < 0.f)
        {
            blendBackfaces++;
            if (blendBackfaces >= maxBackfaces)
            {
                blendBackfaceEarlyOut = true;
                break;
            }
            continue;
        }

        float3 blendRayDirection = DDGIGetProbeRayDirection((int)rayIndex, volume);
        float blendWeight = max(0.f, dot(centerProbeDirection, blendRayDirection));
        blendWeightSum += blendWeight;
    }

    float blendBackfaceRatio = (blendedRayCount > 0u)
        ? ((float)blendBackfaces / (float)blendedRayCount)
        : 0.f;

    uint blendFlags = 0u;
    if (((uint)probeState) == RTXGI_DDGI_PROBE_STATE_INACTIVE)
    {
        blendFlags |= DDGI_PROBE_DEBUG_BLEND_FLAG_INACTIVE_SKIP;
    }
    if (blendBackfaceEarlyOut)
    {
        blendFlags |= DDGI_PROBE_DEBUG_BLEND_FLAG_BACKFACE_EARLY_OUT;
    }

    int numRays = min(volume.probeNumRays, RTXGI_DDGI_NUM_FIXED_RAYS);
    float closestBackfaceDistance = 1e27f;
    float closestFrontfaceDistance = 1e27f;
    float farthestFrontfaceDistance = 0.f;
    uint backfaceCount = 0;
    for (int rayIndex = 0; rayIndex < numRays; rayIndex++)
    {
        int3 rayTexel = DDGIGetRayDataTexelCoords(rayIndex, (int)probeIndex, volume);
        float hitDistance = DDGILoadProbeRayDistance(RayData, rayTexel, volume);
        if (hitDistance < 0.f)
        {
            backfaceCount++;
            hitDistance = hitDistance * -5.f;
            if (hitDistance < closestBackfaceDistance) closestBackfaceDistance = hitDistance;
        }
        else
        {
            if (hitDistance < closestFrontfaceDistance) closestFrontfaceDistance = hitDistance;
            if (hitDistance > farthestFrontfaceDistance) farthestFrontfaceDistance = hitDistance;
        }
    }

    uint segmentStride = maxProbeRecordsPerPass;
    uint baseIndex = volumeBase
        + (DDGI_PROBE_DEBUG_PASS_INDEX * segmentStride)
        + probeIndex;

    DDGIProbeDebugRecord record = (DDGIProbeDebugRecord)0;
    record.probe_meta = uint4((uint3)probeCoords, probeIndex);
    record.trace_probe = float4(probeWorldPos, probeState);
    record.trace_fixed_ray_dir_kind = float4(fixedRayDir, fixedRayDistance < 0.f ? 1.f : 0.f);
    record.trace_fixed_ray_data = float4(fixedRayRadiance, fixedRayDistance);
    record.trace_random_ray_dir_kind = float4(randomRayDir, randomRayDistance < 0.f ? 1.f : 0.f);
    record.trace_random_ray_data = float4(randomRayRadiance, randomRayDistance);
    record.relocate_stats = float4(closestBackfaceDistance, closestFrontfaceDistance, farthestFrontfaceDistance, (numRays > 0) ? (((float)backfaceCount) / ((float)numRays)) : 0.f);
    record.relocate_offset = float4(probeOffset, length(probeOffset));
    record.classify_stats = uint4(backfaceCount, (uint)numRays, (uint)(((uint)probeState) == RTXGI_DDGI_PROBE_STATE_ACTIVE), blendFlags);
    record.classify_probe = float4(probeWorldPos, probeState);
    record.irradiance_center = float4(irradiance.rgb, distanceMoments.x);
    record.distance_center = float4(distanceMoments.xy, blendWeightSum, blendBackfaceRatio);
    record.packed_state = float4(probeOffset, probeState);

    ProbeDebugOut[baseIndex] = record;
}

#else

void DDGIWriteProbeDebugRecord(
    uint probeIndex,
    uint totalProbes,
    uint volumeIndex,
    DDGIVolumeDescGPU volume,
    RWTexture2DArray<float4> RayData,
    RWTexture2DArray<float4> ProbeIrradiance,
    RWTexture2DArray<float4> ProbeDistance,
    RWTexture2DArray<float4> ProbeData)
{
}

#endif
