/*
* Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "SceneSyncExport.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>

#if defined(API_D3D12) && defined(RTXGI_ENABLE_SCENE_SYNC_EXPORT) && RTXGI_ENABLE_SCENE_SYNC_EXPORT

#include <Windows.h>

#ifdef near
#undef near
#endif

#ifdef far
#undef far
#endif

#define RG_SCENE_IMPL
#include "rg_scene_sync.hpp"

namespace
{
    struct SceneSyncRuntime
    {
        HMODULE module = nullptr;
        bool initialized = false;
    };

    rg_scene::Vec3 MakeVec3(float x, float y, float z)
    {
        rg_scene::Vec3 vec;
        vec.x(x).y(y).z(z);
        return vec;
    }

    rg_scene::Vec2 MakeVec2(float x, float y)
    {
        rg_scene::Vec2 vec;
        vec.x(x).y(y);
        return vec;
    }

    rg_scene::Vec4 MakeVec4(float x, float y, float z, float w)
    {
        rg_scene::Vec4 vec;
        vec.x(x).y(y).z(z).w(w);
        return vec;
    }

    rg_scene::Color MakeLinearColor(float r, float g, float b, float a = 1.f)
    {
        rg_scene::LinearRgba linear;
        linear.red(r).green(g).blue(b).alpha(a);

        rg_scene::Color color;
        auto linearColor = rg_scene::set_LinearRgba(color.as_ref());
        linearColor._0(std::move(linear));
        return color;
    }

    rg_scene::Transform MakeTransformWithQuaternion(float x, float y, float z, float qx, float qy, float qz, float qw)
    {
        rg_scene::Transform transform;
        rg_scene::Quat rotation;
        rotation.x(qx).y(qy).z(qz).w(qw);

        transform
            .translation(MakeVec3(x, y, z))
            .rotation(std::move(rotation))
            .scale(MakeVec3(1.f, 1.f, 1.f));

        return transform;
    }

    rg_scene::Transform MakeTransform(float x, float y, float z)
    {
        return MakeTransformWithQuaternion(x, y, z, 0.f, 0.f, 0.f, 1.f);
    }

    std::array<float, 4> MakeQuaternionFromDirection(float dx, float dy, float dz)
    {
        const float lenSq = (dx * dx) + (dy * dy) + (dz * dz);
        if (lenSq <= 1e-8f)
        {
            return { 0.f, 0.f, 0.f, 1.f };
        }

        const float invLen = 1.f / std::sqrt(lenSq);
        const float nx = dx * invLen;
        const float ny = dy * invLen;
        const float nz = dz * invLen;

        // Rotate +X (scene forward axis) to target direction.
        const float dot = std::clamp(nx, -1.f, 1.f);
        if (dot < -0.9999f)
        {
            return { 0.f, 0.f, 1.f, 0.f };
        }

        const float ax = 0.f;
        const float ay = -nz;
        const float az = ny;
        const float aw = 1.f + dot;

        const float qLenSq = (ax * ax) + (ay * ay) + (az * az) + (aw * aw);
        if (qLenSq <= 1e-8f)
        {
            return { 0.f, 0.f, 0.f, 1.f };
        }

        const float invQLen = 1.f / std::sqrt(qLenSq);
        return { ax * invQLen, ay * invQLen, az * invQLen, aw * invQLen };
    }

    rg_scene::Transform MakeTransformFromDirection(float x, float y, float z, float dx, float dy, float dz)
    {
        const std::array<float, 4> q = MakeQuaternionFromDirection(dx, dy, dz);
        return MakeTransformWithQuaternion(x, y, z, q[0], q[1], q[2], q[3]);
    }

    rg_scene::SceneCoordSys MakeSceneCoordSys()
    {
        rg_scene::SceneCoordSys coord;

        // UE convention: centimeters, Z-up, and X-forward / Y-right.
        coord.unit_scale(0.01f);
        coord.right(MakeVec3(0.f, 1.f, 0.f));
        coord.up(MakeVec3(0.f, 0.f, 1.f));
        coord.forward(MakeVec3(1.f, 0.f, 0.f));

        return coord;
    }

    std::string SanitizeName(const std::string& input, const std::string& fallback)
    {
        std::string out;
        out.reserve(input.size());

        for (char ch : input)
        {
            if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-')
            {
                out.push_back(ch);
            }
            else
            {
                out.push_back('_');
            }
        }

        if (out.empty()) return fallback;
        return out;
    }

    rg_scene::ColorTexture MakeLinearRgbaTexture(float r, float g, float b, float a)
    {
        rg_scene::ColorTexture texture;
        auto linear = rg_scene::set_LinearRgba(texture.as_ref());
        linear._0(MakeVec4(r, g, b, a));
        return texture;
    }

    rg_scene::ColorMaterialInput MakeColorInput(float r, float g, float b, float a)
    {
        rg_scene::ColorMaterialInput input;
        input.texture(MakeLinearRgbaTexture(r, g, b, a));
        return input;
    }

    rg_scene::Buffer<rg_scene::Vec3> MakeVec3Buffer(const std::vector<Graphics::Vertex>& vertices, const bool useNormal)
    {
        rg_scene::Vec<rg_scene::Vec3> values;
        for (const Graphics::Vertex& vertex : vertices)
        {
            if (useNormal)
            {
                values.push(MakeVec3(vertex.normal.x, vertex.normal.y, vertex.normal.z));
            }
            else
            {
                values.push(MakeVec3(vertex.position.x, vertex.position.y, vertex.position.z));
            }
        }

        rg_scene::Buffer<rg_scene::Vec3>::Raw raw;
        raw._0(std::move(values));
        return raw;
    }

    rg_scene::Buffer<rg_scene::Vec4> MakeTangentBuffer(const std::vector<Graphics::Vertex>& vertices, const bool tangentX)
    {
        rg_scene::Vec<rg_scene::Vec4> values;

        for (const Graphics::Vertex& vertex : vertices)
        {
            if (tangentX)
            {
                values.push(MakeVec4(vertex.tangent.x, vertex.tangent.y, vertex.tangent.z, vertex.tangent.w));
                continue;
            }

            const float nx = vertex.normal.x;
            const float ny = vertex.normal.y;
            const float nz = vertex.normal.z;

            const float tx = vertex.tangent.x;
            const float ty = vertex.tangent.y;
            const float tz = vertex.tangent.z;
            const float tw = vertex.tangent.w;

            float bx = (ny * tz - nz * ty) * tw;
            float by = (nz * tx - nx * tz) * tw;
            float bz = (nx * ty - ny * tx) * tw;

            const float lenSq = (bx * bx) + (by * by) + (bz * bz);
            if (lenSq > 0.f)
            {
                const float invLen = 1.f / std::sqrt(lenSq);
                bx *= invLen;
                by *= invLen;
                bz *= invLen;
            }

            values.push(MakeVec4(bx, by, bz, 1.f));
        }

        rg_scene::Buffer<rg_scene::Vec4>::Raw raw;
        raw._0(std::move(values));
        return raw;
    }

    rg_scene::Buffer<rg_scene::u32> MakeIndexBuffer(const std::vector<rg_scene::u32>& indices)
    {
        rg_scene::Vec<rg_scene::u32> values;

        for (rg_scene::u32 index : indices)
        {
            values.push(index);
        }

        rg_scene::Buffer<rg_scene::u32>::Raw raw;
        raw._0(std::move(values));
        return raw;
    }

    rg_scene::Buffer<rg_scene::Vec2> MakeUV0Buffer(const std::vector<Graphics::Vertex>& vertices)
    {
        rg_scene::Vec<rg_scene::Vec2> values;
        for (const Graphics::Vertex& vertex : vertices)
        {
            values.push(MakeVec2(vertex.uv0.x, vertex.uv0.y));
        }

        rg_scene::Buffer<rg_scene::Vec2>::Raw raw;
        raw._0(std::move(values));
        return raw;
    }

    rg_scene::Vec<rg_scene::Resource<rg_scene::Material>> MakeSectionMaterialRefs(
        const std::vector<int>& primitiveMaterialIndices,
        const std::vector<std::string>& materialPaths)
    {
        rg_scene::Vec<rg_scene::Resource<rg_scene::Material>> refs;

        for (int materialIndex : primitiveMaterialIndices)
        {
            if (materialIndex >= 0 && materialIndex < static_cast<int>(materialPaths.size()))
            {
                rg_scene::Resource<rg_scene::Material>::Reference materialRef;
                materialRef.path(materialPaths[materialIndex]);
                refs.push(std::move(materialRef));
            }
            else
            {
                refs.push(rg_scene::Resource<rg_scene::Material>::Nil());
            }
        }

        return refs;
    }

    bool EnsureSceneSyncLoaded(const std::string& dllPath, std::ofstream& log)
    {
        static SceneSyncRuntime runtime;
        if (runtime.initialized) return true;

        runtime.module = LoadLibraryA(dllPath.c_str());
        if (runtime.module == nullptr)
        {
            log << "[SceneSync] Failed to load DLL: " << dllPath << "\n";
            return false;
        }

        using LoadSymbolsFn = void(*)(Rg_sceneSymbols*);
        auto loadSymbols = reinterpret_cast<LoadSymbolsFn>(GetProcAddress(runtime.module, "rg_scene_load_symbols"));
        if (loadSymbols == nullptr)
        {
            log << "[SceneSync] Missing exported symbol rg_scene_load_symbols in: " << dllPath << "\n";
            return false;
        }

        Rg_sceneSymbols symbols = {};
        loadSymbols(&symbols);
        rg_scene_init_symbols(symbols);

        rg_init(
            3,
            true,
            nullptr,
            [](const char* Msg, uint64_t Len)
            {
				printf("[SceneSync] %s", Msg);
            },
            []() {}
        );

        runtime.initialized = true;
        return true;
    }

    void FillDDGIVolumes(const std::vector<rtxgi::DDGIVolumeDesc>& volumeDescs, rg_scene::Scene& scene)
    {
        rg_scene::Vec<rg_scene::DdgiVolume> ddgiVolumes;

        for (size_t volumeIndex = 0; volumeIndex < volumeDescs.size(); volumeIndex++)
        {
            const rtxgi::DDGIVolumeDesc& volumeDesc = volumeDescs[volumeIndex];

            float halfX = 0.5f * std::max(0, volumeDesc.probeCounts.x - 1) * volumeDesc.probeSpacing.x;
            float halfY = 0.5f * std::max(0, volumeDesc.probeCounts.y - 1) * volumeDesc.probeSpacing.y;
            float halfZ = 0.5f * std::max(0, volumeDesc.probeCounts.z - 1) * volumeDesc.probeSpacing.z;

            float minSpacing = std::min(volumeDesc.probeSpacing.x, std::min(volumeDesc.probeSpacing.y, volumeDesc.probeSpacing.z));
            minSpacing = std::max(minSpacing, 0.f);

            std::string volumeName = (volumeDesc.name != nullptr && volumeDesc.name[0] != '\0')
                ? std::string(volumeDesc.name)
                : ("ddgi_volume_" + std::to_string(volumeIndex));

            rg_scene::DdgiVolume ddgiVolume;
            ddgiVolume
                .name(volumeName)
                .bounds_min(MakeVec3(volumeDesc.origin.x - halfX, volumeDesc.origin.y - halfY, volumeDesc.origin.z - halfZ))
                .bounds_max(MakeVec3(volumeDesc.origin.x + halfX, volumeDesc.origin.y + halfY, volumeDesc.origin.z + halfZ))
                .probe_count_x(static_cast<rg_scene::u32>(std::max(0, volumeDesc.probeCounts.x)))
                .probe_count_y(static_cast<rg_scene::u32>(std::max(0, volumeDesc.probeCounts.y)))
                .probe_count_z(static_cast<rg_scene::u32>(std::max(0, volumeDesc.probeCounts.z)))
                .blend_distance(minSpacing)
                .blend_distance_black(0.f)
                .irradiance_scalar(1.f);

            ddgiVolumes.push(std::move(ddgiVolume));
        }

        scene.ddgi_volumes(std::move(ddgiVolumes));
    }

    void FillMaterials(const Scenes::Scene& sourceScene, rg_scene::Scene& targetScene, std::vector<std::string>& materialPaths)
    {
        materialPaths.clear();
        materialPaths.resize(sourceScene.materials.size());

        rg_scene::Vec<rg_scene::Resource<rg_scene::Material>> materials;

        for (size_t materialIndex = 0; materialIndex < sourceScene.materials.size(); materialIndex++)
        {
            const Scenes::Material& sourceMaterial = sourceScene.materials[materialIndex];

            std::string materialName = sourceMaterial.name.empty()
                ? ("material_" + std::to_string(materialIndex))
                : sourceMaterial.name;
            std::string materialPath = "materials/" + SanitizeName(materialName, "material") + "_" + std::to_string(materialIndex) + ".ron";

            rg_scene::Material material;
            material
                .name(materialName)
                .two_sided(sourceMaterial.data.doubleSided != 0);

            rg_scene::DiffuseMaterialAttributes diffuse;
            diffuse.reflectance(MakeColorInput(
                sourceMaterial.data.albedo.x,
                sourceMaterial.data.albedo.y,
                sourceMaterial.data.albedo.z,
                sourceMaterial.data.opacity));

            rg_scene::MaterialAttributes::Diffuse attributes;
            attributes._0(std::move(diffuse));
            material.attributes(std::move(attributes));

            rg_scene::Resource<rg_scene::Material>::Inline materialRes;
            materialRes.path(materialPath);
            materialRes.content(std::move(material));

            materialPaths[materialIndex] = materialPath;
            materials.push(std::move(materialRes));
        }

        targetScene.materials(std::move(materials));
    }

    void FillMeshesAndActors(
        const Scenes::Scene& sourceScene,
        const std::vector<std::string>& materialPaths,
        rg_scene::Scene& targetScene,
        rg_scene::Vec<rg_scene::Actor>& actors)
    {
        std::vector<std::string> meshPaths;
        meshPaths.resize(sourceScene.meshes.size());

        rg_scene::Vec<rg_scene::Resource<rg_scene::Mesh>> meshes;

        for (size_t meshIndex = 0; meshIndex < sourceScene.meshes.size(); meshIndex++)
        {
            const Scenes::Mesh& sourceMesh = sourceScene.meshes[meshIndex];

            std::string meshName = sourceMesh.name.empty()
                ? ("mesh_" + std::to_string(meshIndex))
                : sourceMesh.name;
            std::string meshPath = "meshes/" + SanitizeName(meshName, "mesh") + "_" + std::to_string(meshIndex) + ".ron";

            rg_scene::Mesh mesh;
            mesh.name(meshName);

            std::vector<Graphics::Vertex> allVertices;
            std::vector<rg_scene::u32> allIndices;
            std::vector<int> sectionMaterialIndices;

            rg_scene::Vec<rg_scene::MeshSection> sections;

            rg_scene::u32 baseVertex = 0;
            rg_scene::u32 indexCursor = 0;

            for (const Scenes::MeshPrimitive& primitive : sourceMesh.primitives)
            {
                if (primitive.vertices.empty() || primitive.indices.empty()) continue;

                allVertices.insert(allVertices.end(), primitive.vertices.begin(), primitive.vertices.end());

                const rg_scene::u32 startIndex = indexCursor;
                for (uint32_t primitiveIndex : primitive.indices)
                {
                    allIndices.push_back(baseVertex + static_cast<rg_scene::u32>(primitiveIndex));
                    indexCursor++;
                }

                rg_scene::MeshSection section;
                section.start_index(startIndex);
                section.num_tris(static_cast<rg_scene::u32>(primitive.indices.size() / 3));
                sections.push(std::move(section));

                sectionMaterialIndices.push_back(primitive.material);
                baseVertex += static_cast<rg_scene::u32>(primitive.vertices.size());
            }

            if (!allVertices.empty() && !allIndices.empty())
            {
                mesh.positions(MakeVec3Buffer(allVertices, false));
                mesh.normals(MakeVec3Buffer(allVertices, true));
                mesh.tangents_x(MakeTangentBuffer(allVertices, true));
                mesh.tangents_y(MakeTangentBuffer(allVertices, false));
                mesh.triangles(MakeIndexBuffer(allIndices));

                rg_scene::Vec<rg_scene::Buffer<rg_scene::Vec2>> uvChannels;
                uvChannels.push(MakeUV0Buffer(allVertices));
                mesh.uv_coords(std::move(uvChannels));

                mesh.sections(std::move(sections));
                mesh.section_materials(MakeSectionMaterialRefs(sectionMaterialIndices, materialPaths));

                rg_scene::Resource<rg_scene::Mesh>::Inline meshRes;
                meshRes.path(meshPath);
                meshRes.content(std::move(mesh));

                meshPaths[meshIndex] = meshPath;
                meshes.push(std::move(meshRes));
            }
            else
            {
                meshPaths[meshIndex].clear();
            }
        }

        targetScene.meshes(std::move(meshes));

        for (size_t instanceIndex = 0; instanceIndex < sourceScene.instances.size(); instanceIndex++)
        {
            const Scenes::MeshInstance& instance = sourceScene.instances[instanceIndex];

            if (instance.meshIndex < 0 || instance.meshIndex >= static_cast<int>(meshPaths.size())) continue;
            if (meshPaths[instance.meshIndex].empty()) continue;

            std::string actorName = instance.name.empty()
                ? ("mesh_instance_" + std::to_string(instanceIndex))
                : instance.name;

            rg_scene::StaticMeshComponent staticMesh;
            staticMesh.name(actorName + "_static_mesh");

            rg_scene::Resource<rg_scene::Mesh>::Reference meshRef;
            meshRef.path(meshPaths[instance.meshIndex]);
            staticMesh.mesh(std::move(meshRef));

            const float tx = instance.transform[0][3];
            const float ty = instance.transform[1][3];
            const float tz = instance.transform[2][3];
            staticMesh.transform(MakeTransform(tx, ty, tz));
            staticMesh.local_transform(MakeTransform(tx, ty, tz));

            const Scenes::Mesh& sourceMesh = sourceScene.meshes[instance.meshIndex];
            std::vector<int> primitiveMaterialIndices;
            primitiveMaterialIndices.reserve(sourceMesh.primitives.size());
            for (const Scenes::MeshPrimitive& primitive : sourceMesh.primitives)
            {
                primitiveMaterialIndices.push_back(primitive.material);
            }
            staticMesh.section_materials(MakeSectionMaterialRefs(primitiveMaterialIndices, materialPaths));

            rg_scene::Component::StaticMesh meshComponent;
            meshComponent._0(std::move(staticMesh));

            rg_scene::Vec<rg_scene::Component> components;
            components.push(std::move(meshComponent));

            rg_scene::Actor actor;
            actor
                .label(actorName)
                .name(actorName)
                .hidden(false)
                .components(std::move(components));

            actors.push(std::move(actor));
        }
    }

    void FillLightActors(const Scenes::Scene& sourceScene, rg_scene::Vec<rg_scene::Actor>& actors)
    {
        constexpr float kDegToRad = 3.14159265358979323846f / 180.f;

        for (size_t lightIndex = 0; lightIndex < sourceScene.lights.size(); lightIndex++)
        {
            const Scenes::Light& srcLight = sourceScene.lights[lightIndex];

            rg_scene::Actor actor;
            rg_scene::Vec<rg_scene::Component> components;

            std::string baseName = srcLight.name.empty() ? ("light_" + std::to_string(lightIndex)) : srcLight.name;

            actor
                .label(baseName)
                .name(baseName)
                .hidden(false);

            if (srcLight.type == ELightType::POINT)
            {
                rg_scene::PointLightComponent light;
                light
                    .name(baseName)
                    .position(MakeVec3(srcLight.data.position.x, srcLight.data.position.y, srcLight.data.position.z))
                    .color(MakeLinearColor(srcLight.data.color.x, srcLight.data.color.y, srcLight.data.color.z))
                    .intensity(srcLight.data.power)
                    .attenuation_radius(srcLight.data.radius);

                rg_scene::Component::PointLight component;
                component._0(std::move(light));
                components.push(std::move(component));
            }
            else if (srcLight.type == ELightType::SPOT)
            {
                const float outerAngle = std::max(srcLight.data.penumbraAngle * kDegToRad, 1e-4f);
                const float innerAngle = std::clamp(srcLight.data.umbraAngle * kDegToRad, 0.f, outerAngle);

                rg_scene::SpotLightComponent light;
                light
                    .name(baseName)
                    .transform(MakeTransformFromDirection(
                        srcLight.data.position.x,
                        srcLight.data.position.y,
                        srcLight.data.position.z,
                        srcLight.data.direction.x,
                        srcLight.data.direction.y,
                        srcLight.data.direction.z))
                    .color(MakeLinearColor(srcLight.data.color.x, srcLight.data.color.y, srcLight.data.color.z))
                    .intensity(srcLight.data.power)
                    .attenuation_radius(srcLight.data.radius)
                    .inner_angle(innerAngle)
                    .outer_angle(outerAngle);

                rg_scene::Component::SpotLight component;
                component._0(std::move(light));
                components.push(std::move(component));
            }
            else if (srcLight.type == ELightType::DIRECTIONAL)
            {
                const float dirLenSq =
                    (srcLight.data.direction.x * srcLight.data.direction.x) +
                    (srcLight.data.direction.y * srcLight.data.direction.y) +
                    (srcLight.data.direction.z * srcLight.data.direction.z);

                float dx = 1.f;
                float dy = 0.f;
                float dz = 0.f;
                if (dirLenSq > 1e-8f)
                {
                    const float invLen = 1.f / std::sqrt(dirLenSq);
                    dx = srcLight.data.direction.x * invLen;
                    dy = srcLight.data.direction.y * invLen;
                    dz = srcLight.data.direction.z * invLen;
                }

                rg_scene::DirectionalLightComponent light;
                light
                    .name(baseName)
                    .intensity(srcLight.data.power)
                    .color(MakeLinearColor(srcLight.data.color.x, srcLight.data.color.y, srcLight.data.color.z))
                    .dir(MakeVec3(dx, dy, dz));

                rg_scene::Component::DirectionalLight component;
                component._0(std::move(light));
                components.push(std::move(component));
            }

            if (components.len() > 0)
            {
                actor.components(std::move(components));
                actors.push(std::move(actor));
            }
        }
    }
}

namespace SceneSyncExport
{
    bool ExportIfRequested(
        const Configs::Config& config,
        const Scenes::Scene& scene,
        const std::vector<rtxgi::DDGIVolumeDesc>& volumeDescs,
        std::ofstream& log)
    {
        if (!config.ddgi.sceneSyncExportEnabled || !config.ddgi.sceneSyncExportOnStartup) return true;

        if (config.ddgi.sceneSyncExportPath.empty())
        {
            log << "[SceneSync] Export path is empty, skipped.\n";
            return false;
        }

        if (!EnsureSceneSyncLoaded(config.ddgi.sceneSyncDllPath, log))
        {
            return false;
        }

        rg_scene::Scene exportScene;
        exportScene.name(scene.name);

        rg_scene::MeshConfig meshConfig;
        rg_scene::MeshFrontFace::Ccw frontFace;
        meshConfig.front_face(std::move(frontFace));
        exportScene.mesh_config(std::move(meshConfig));
        exportScene.coord_sys(MakeSceneCoordSys());

        FillDDGIVolumes(volumeDescs, exportScene);

        std::vector<std::string> materialPaths;
        FillMaterials(scene, exportScene, materialPaths);

        rg_scene::Vec<rg_scene::Actor> actors;
        FillMeshesAndActors(scene, materialPaths, exportScene, actors);
        FillLightActors(scene, actors);
        exportScene.actors(std::move(actors));

        rg_scene::rg_save_scene(exportScene, config.ddgi.sceneSyncExportPath);
        log << "[SceneSync] Scene exported to: " << config.ddgi.sceneSyncExportPath << "\n";

        return true;
    }
}

#else

namespace SceneSyncExport
{
    bool ExportIfRequested(
        const Configs::Config& config,
        const Scenes::Scene&,
        const std::vector<rtxgi::DDGIVolumeDesc>&,
        std::ofstream& log)
    {
        if (config.ddgi.sceneSyncExportEnabled && config.ddgi.sceneSyncExportOnStartup)
        {
            log << "[SceneSync] Export requested but rg_scene_sync support is unavailable in this build.\n";
            return false;
        }
        return true;
    }
}

#endif
